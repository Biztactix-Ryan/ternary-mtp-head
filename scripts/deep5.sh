#!/bin/bash
# Train the ternary rank-4096 head 5 drafts deep, then test 3, 4 and 5 drafts in the engine on the
# re-packed trunk. Results: logs/deep5-summary.txt
cd ~/ttf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=ttf-home/.venv/bin/python
B=ttf-home/engine-q38/build-cuda/bin
MTP=ttf-home/models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
TTF=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
GPU=${1:-0}
train() { # seq-len chunk
  CUDA_VISIBLE_DEVICES=$GPU $PY mtp/train_mtp.py --mtp $MTP --dump data/dump-ttf,data/dump-ttf2 --emb-from $TTF --half-state \
    --eval-seqs 150 --eval-every 100 --warmup 30 --lr 5e-5 --init out/h2h-float/best.pt --ttf --rank 4096 --rank-kv 1024 \
    --init-ttf out/nx-ttf4096-s2long/best.pt --steps 500 --unroll 5 --unroll-weights 1,0.7,0.5,0.4,0.3 \
    --seq-len $1 --chunk $2 --out out/deep5 > logs/deep5-train.log 2>&1
}
train 768 768 || train 512 512
$PY mtp/build_infile.py out/trunk-tq1p.gguf out/tmp-deep5.gguf --ttf-ckpt out/deep5/best.pt > logs/deep5-build.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $PY mtp/repack_tq1p.py out/tmp-deep5.gguf out/ttf-27b-mtp-deep5-tq1p.gguf >> logs/deep5-build.log 2>&1
rm -f out/tmp-deep5.gguf
rm -f logs/deep5-engine.log
for n in 3 4 5; do
  CUDA_VISIBLE_DEVICES=$GPU $B/llama-server -m out/ttf-27b-mtp-deep5-tq1p.gguf -ngl 99 -c 16384 -fa on --host 127.0.0.1 --port 8097 \
     --spec-type draft-mtp --spec-draft-n-max $n > logs/deep5-server-n$n.log 2>&1 &
  pid=$!
  for i in $(seq 150); do curl -sf http://127.0.0.1:8097/health > /dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
  python3 mtp/bench12.py 8097 deep5-n$n >> logs/deep5-engine.log 2>&1
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
done
{ echo "== head trained 5 drafts deep $(date '+%F %T')"; grep -E "^eval" logs/deep5-train.log | sed -n '1p;$p' | cut -c1-230
  grep -E "^done|Error" logs/deep5-train.log | tail -1; grep TOTAL logs/deep5-engine.log; } > logs/deep5-summary.txt
echo DEEP5_DONE >> logs/deep5-summary.txt
