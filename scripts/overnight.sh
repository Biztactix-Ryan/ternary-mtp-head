#!/bin/bash
# Unattended next stage. Waits for the bulk dump (AFTER_BATCH2_DONE), then on the combined data:
#   GPU1: float reference (stage 1) -> history-weight ablation (0 / 0.2 / 1.0) -> context-length check
#         -> float stage 2 (multi-draft) -> engine tests
#   GPU0: ternary rank-4096 stage 1 -> ternary stage 2 -> ternary rank-8192 (about 260 MiB) stage 1
# Every arm starts from the same checkpoint (out/h2h-float/best.pt). Results: logs/overnight-summary.txt
cd ~/ttf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=ttf-home/.venv/bin/python
B=ttf-home/engine-q38/build-cuda/bin
MTP=ttf-home/models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
TTF=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
DUMP=data/dump-ttf,data/dump-ttf2
INIT=out/h2h-float/best.pt
L=logs/overnight.log
say() { echo "$(date '+%F %T') $*" >> $L; }

while ! grep -q AFTER_BATCH2_DONE logs/after-batch2.log 2>/dev/null; do sleep 60; done
say "dump finished: $(tail -4 logs/after-batch2.log | tr '\n' ' ')"

train() { # gpu name args...
  gpu=$1; name=$2; shift 2
  say "start $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu $PY mtp/train_mtp.py --mtp $MTP --dump $DUMP --emb-from $TTF --half-state \
      --eval-seqs 150 --eval-every 100 --warmup 50 --lr 1e-4 --out out/on-$name "$@" > logs/on-$name.log 2>&1
  rc=$?
  say "end $name rc=$rc: $(grep -E '^done|Error' logs/on-$name.log | tail -1 | cut -c1-120)"
  return $rc
}

serve_bench() { # gpu label model extra...
  gpu=$1; label=$2; model=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $model -ngl 99 -c 8192 -fa on --host 127.0.0.1 --port 8097 "$@" > logs/on-server-$label.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do
    curl -sf http://127.0.0.1:8097/health > /dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then python3 mtp/bench12.py 8097 $label >> logs/on-engine.log 2>&1
  else echo "$label: server failed to start: $(grep -iE 'error|expected|abort' logs/on-server-$label.log | tail -2)" >> logs/on-engine.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

engine() { # gpu name ckpt-flag ckpt
  gpu=$1; name=$2
  [ -f "$4" ] || { say "engine $name: no checkpoint $4"; return; }
  $PY mtp/build_infile.py $TTF out/on-$name.gguf $3 $4 > logs/on-build-$name.log 2>&1 || { say "engine $name: build failed"; return; }
  for n in 1 2 3; do serve_bench $gpu $name-n$n out/on-$name.gguf --spec-type draft-mtp --spec-draft-n-max $n; done
  say "engine $name done: $(grep "$name-n. TOTAL" logs/on-engine.log | cut -c1-90 | tr '\n' ';')"
}

# ---------------- GPU 1 chain ----------------
(
  train 1 float-s1 --init $INIT --steps 1500
  for w in 0 0.2 1; do train 1 hist-w$w --init $INIT --steps 600 --prompt-weight $w; rm -f out/on-hist-w$w/best.pt; done
  say "start context-length check"
  CUDA_VISIBLE_DEVICES=1 $PY mtp/mtp_head.py eval --mtp $MTP --emb-from $TTF --ckpt $INIT --dump data/dump-ttf2 --held-only --by-prefix \
      > logs/on-context.log 2>&1
  train 1 float-s2 --init out/on-float-s1/best.pt --steps 400 --unroll 3 --seq-len 1024 --chunk 1024 --lr 5e-5
  serve_bench 1 plain $TTF
  engine 1 float-s1 --float-ckpt out/on-float-s1/best.pt
  engine 1 float-s2 --float-ckpt out/on-float-s2/best.pt
  # ternary heads, as soon as the other card has produced them
  while [ ! -f logs/on-gpu0-s2.done ]; do sleep 60; done
  engine 1 ttf4096-s1 --ttf-ckpt out/on-ttf4096-s1/best.pt
  engine 1 ttf4096-s2 --ttf-ckpt out/on-ttf4096-s2/best.pt
) &

# ---------------- GPU 0 chain ----------------
(
  train 0 ttf4096-s1 --init $INIT --ttf --rank 4096 --rank-kv 1024 --steps 1500
  train 0 ttf4096-s2 --init $INIT --ttf --rank 4096 --rank-kv 1024 --init-ttf out/on-ttf4096-s1/best.pt \
        --steps 400 --unroll 3 --seq-len 1024 --chunk 1024 --lr 5e-5 \
  || train 0 ttf4096-s2 --init $INIT --ttf --rank 4096 --rank-kv 1024 --init-ttf out/on-ttf4096-s1/best.pt \
        --steps 400 --unroll 3 --seq-len 640 --chunk 512 --lr 5e-5
  touch logs/on-gpu0-s2.done
  train 0 ttf8192-s1 --init $INIT --ttf --rank 8192 --rank-kv 2048 --steps 1000 --chunk 1024 \
  || train 0 ttf8192-s1 --init $INIT --ttf --rank 8192 --rank-kv 2048 --steps 1000 --chunk 512 --seq-len 1024
  rm -f out/on-ttf8192-s1/last.pt
) &
wait

# the rank-8192 head in the engine, once both chains are finished
( engine 0 ttf8192-s1 --ttf-ckpt out/on-ttf8192-s1/best.pt )

{
  echo "== overnight summary $(date '+%F %T')"
  echo; echo "-- held-out results (last eval line of each run)"
  for f in logs/on-float-s1.log logs/on-ttf4096-s1.log logs/on-ttf8192-s1.log logs/on-hist-w0.log logs/on-hist-w0.2.log logs/on-hist-w1.log \
           logs/on-float-s2.log logs/on-ttf4096-s2.log; do
    [ -f $f ] && echo "$(basename $f .log): $(grep -E '^sequences' $f | head -1 | cut -c1-110)" && \
      echo "   first: $(grep -E '^eval' $f | head -1 | cut -c12-170)" && echo "   last:  $(grep -E '^eval' $f | tail -1 | cut -c12-170)" && \
      echo "   $(grep -E '^ternary|^done|Error' $f | tr '\n' ' ' | cut -c1-230)"
  done
  echo; echo "-- context-length check (head trained on short contexts, scored per source)"; grep -vE "Warning|from_numpy" logs/on-context.log
  echo; echo "-- engine, 12 prompts"; grep TOTAL logs/on-engine.log; grep "failed" logs/on-engine.log
  echo; ls -l out/on-*.gguf 2>/dev/null | awk '{printf "%.0f MiB  %s\n", $5/1048576, $9}'
} > logs/overnight-summary.txt 2>&1
say "ALL DONE"
