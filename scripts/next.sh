#!/bin/bash
# Follow-up experiments after the overnight run (unattended). Results: logs/next-summary.txt
#   GPU1: confidence-threshold sweep in the engine (--spec-draft-p-min) -> history-only training test
#         -> longer stage 2 for the float head -> engine tests
#   GPU0: longer stage 2 for the ternary rank-4096 head -> engine tests
cd ~/ttf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=ttf-home/.venv/bin/python
B=ttf-home/engine-q38/build-cuda/bin
MTP=ttf-home/models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
TTF=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
DUMP=data/dump-ttf,data/dump-ttf2
INIT=out/h2h-float/best.pt
L=logs/next.log
say() { echo "$(date '+%F %T') $*" >> $L; }

train() { # gpu name args...
  gpu=$1; name=$2; shift 2
  say "start $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu $PY mtp/train_mtp.py --mtp $MTP --dump $DUMP --emb-from $TTF --half-state \
      --eval-seqs 150 --eval-every 100 --warmup 50 --lr 1e-4 --out out/nx-$name "$@" > logs/nx-$name.log 2>&1
  say "end $name rc=$?: $(grep -E '^done|Error' logs/nx-$name.log | tail -1 | cut -c1-120)"
}

serve_bench() { # gpu port label model extra...
  gpu=$1; port=$2; label=$3; model=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $model -ngl 99 -c 8192 -fa on --host 127.0.0.1 --port $port "$@" > logs/nx-server-$label.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do
    curl -sf http://127.0.0.1:$port/health > /dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then python3 mtp/bench12.py $port $label >> logs/nx-engine.log 2>&1
  else echo "$label: server failed to start: $(grep -iE 'error|expected|abort|invalid' logs/nx-server-$label.log | tail -2)" >> logs/nx-engine.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

# ---------------- GPU 1 ----------------
(
  say "start p-min sweep"
  M=out/on-ttf4096-s2.gguf
  for n in 2 3; do for p in 0.3 0.5 0.7 0.9; do
    serve_bench 1 8097 pmin-n$n-p$p $M --spec-type draft-mtp --spec-draft-n-max $n --spec-draft-p-min $p
  done; done
  say "end p-min sweep"
  train 1 hist-only --init $INIT --steps 600 --gen-weight 0 --prompt-weight 1; rm -f out/nx-hist-only/best.pt
  train 1 float-s2long --init out/on-float-s2/best.pt --steps 1200 --unroll 3 --seq-len 1024 --chunk 1024 --lr 5e-5
  $PY mtp/build_infile.py $TTF out/nx-float-s2long.gguf --float-ckpt out/nx-float-s2long/best.pt > /dev/null 2>&1
  for n in 1 2 3; do serve_bench 1 8097 float-s2long-n$n out/nx-float-s2long.gguf --spec-type draft-mtp --spec-draft-n-max $n; done
) &

# ---------------- GPU 0 ----------------
(
  train 0 ttf4096-s2long --init $INIT --ttf --rank 4096 --rank-kv 1024 --init-ttf out/on-ttf4096-s2/best.pt \
        --steps 1200 --unroll 3 --seq-len 1024 --chunk 1024 --lr 5e-5
  $PY mtp/build_infile.py $TTF out/nx-ttf4096-s2long.gguf --ttf-ckpt out/nx-ttf4096-s2long/best.pt > /dev/null 2>&1
  for n in 1 2 3; do serve_bench 0 8098 ttf4096-s2long-n$n out/nx-ttf4096-s2long.gguf --spec-type draft-mtp --spec-draft-n-max $n; done
) &
wait

{
  echo "== follow-up summary $(date '+%F %T')"
  echo; echo "-- confidence threshold sweep (ternary rank-4096 stage-2 head; reference without threshold: n2 42.0, n3 37.8, n1 43.9 tok/s)"
  grep "pmin.*TOTAL\|pmin.*failed" logs/nx-engine.log
  echo; echo "-- history-only training (generated weight 0, history weight 1), scored on generated positions; reference 0/0.2/1.0 history with generated on: 64.9 / 65.6 / 65.9"
  grep -E "^eval" logs/nx-hist-only.log | sed -n '1p;$p' | cut -c1-120
  echo; echo "-- longer stage 2"
  for f in float-s2long ttf4096-s2long; do echo "$f:"; grep -E "^eval" logs/nx-$f.log | sed -n '1p;$p' | cut -c1-175; grep -E "^done|Error" logs/nx-$f.log | tail -1; done
  echo; echo "-- engine, 12 prompts (plain 35.3 tok/s)"; grep "s2long.*TOTAL\|s2long.*failed" logs/nx-engine.log
} > logs/next-summary.txt 2>&1
say "ALL DONE"
