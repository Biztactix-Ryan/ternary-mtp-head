#!/bin/bash
# Head to head in the engine: ternary trunk with an in-file ternary head vs an in-file F16 head.
# usage: h2h-engine.sh [gpu]
GPU=${1:-1}; export CUDA_VISIBLE_DEVICES=$GPU
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
TTF=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
PORT=8097

$PY mtp/build_infile.py $TTF out/ttf-27b-mtp-h2h-float.gguf --float-ckpt out/h2h-float/best.pt | tail -1
$PY mtp/build_infile.py $TTF out/ttf-27b-mtp-h2h-ternary.gguf --ttf-ckpt out/h2h-ttf-r1024/best.pt | tail -1
ls -l $TTF out/ttf-27b-mtp-h2h-float.gguf out/ttf-27b-mtp-h2h-ternary.gguf | awk '{printf "%.0f MiB  %s\n", $5/1048576, $9}'

run() { # label, model, extra args...
  label=$1; model=$2; shift 2
  $B/llama-server -m $model -ngl 99 -c 8192 -fa on --host 127.0.0.1 --port $PORT "$@" > logs/server-$label.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do
    curl -sf http://127.0.0.1:$PORT/health > /dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then
    python3 mtp/bench3.py $PORT $label
    echo "$label VRAM $(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader)"
  else
    echo "$label: server failed to start"; grep -iE "error|fail|assert|abort|expected" logs/server-$label.log | tail -8; tail -3 logs/server-$label.log
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
run plain $TTF
for n in 1 2 3; do
  run float-n$n   out/ttf-27b-mtp-h2h-float.gguf   --spec-type draft-mtp --spec-draft-n-max $n
  run ternary-n$n out/ttf-27b-mtp-h2h-ternary.gguf --spec-type draft-mtp --spec-draft-n-max $n
done
grep -h "TTF side tensors" logs/server-ternary-n1.log logs/server-float-n1.log
