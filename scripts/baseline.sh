#!/bin/bash
# Section 4 baseline: checksums, llama-bench, plain decode and stock MTP head on the ternary model.
# usage: baseline.sh [gpu]
GPU=${1:-1}
cd ~/ttf/ttf-home
B=engine-q38/build-cuda/bin
M=models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
MTP=models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
export CUDA_VISIBLE_DEVICES=$GPU
PORT=8097

echo "== sha256"
sha256sum $M $MTP | cut -c1-64

echo "== llama-bench"
$B/llama-bench -m $M -p 512 -n 128 -ngl 99 2>/dev/null | tail -4

run() { # label, extra server args...
  label=$1; shift
  $B/llama-server -m $M -ngl 99 -c 8192 -fa on --port $PORT "$@" > ~/ttf/logs/server-$label.log 2>&1 &
  pid=$!
  for i in $(seq 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
  python3 ~/ttf/mtp/bench3.py $PORT $label
  kill $pid; wait $pid 2>/dev/null
  grep -i "draft acceptance" ~/ttf/logs/server-$label.log | tail -3
}
mkdir -p ~/ttf/logs
echo "== plain";        run plain
echo "== stock MTP n3"; run mtp3 -md $MTP --spec-type draft-mtp --spec-draft-n-max 3
echo "== stock MTP n2"; run mtp2 -md $MTP --spec-type draft-mtp --spec-draft-n-max 2
