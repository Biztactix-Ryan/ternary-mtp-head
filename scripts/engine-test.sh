#!/bin/bash
# Section 10: export, round-trip check on the control model, then the real test on the ternary model.
# usage: engine-test.sh [gpu]
export CUDA_VISIBLE_DEVICES=${1:-1}
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
TTF=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
STOCK=ttf-home/models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
PORT=8097

$PY mtp/export_mtp.py $STOCK out/mtp-ternary-ft.gguf --ckpt out/s1-full/best.pt | tail -1
$PY mtp/export_mtp.py $STOCK out/mtp-ternary-input.gguf --ckpt out/s1-input/best.pt | tail -1

run() { # label, model, extra server args...
  label=$1; model=$2; shift 2
  $B/llama-server -m $model -ngl 99 -c 8192 -fa on --host 127.0.0.1 --port $PORT "$@" > logs/server-$label.log 2>&1 &
  pid=$!
  for i in $(seq 150); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
  python3 mtp/bench3.py $PORT $label
  kill $pid; wait $pid 2>/dev/null
}
echo "== round trip on control (Q4_K_XL), 3 drafts"
run q4-plain      $Q4
run q4-stock      $Q4 -md $STOCK --spec-type draft-mtp --spec-draft-n-max 3
run q4-stock-f16  $Q4 -md out/mtp-stock-f16.gguf --spec-type draft-mtp --spec-draft-n-max 3
echo "== ternary, fine-tuned head"
run ttf-plain $TTF
for n in 1 2 3; do
  run ttf-stock-n$n $TTF -md $STOCK --spec-type draft-mtp --spec-draft-n-max $n
  run ttf-ft-n$n    $TTF -md out/mtp-ternary-ft.gguf --spec-type draft-mtp --spec-draft-n-max $n
done
echo "== ternary, input-only variant"
for n in 1 2; do run ttf-input-n$n $TTF -md out/mtp-ternary-input.gguf --spec-type draft-mtp --spec-draft-n-max $n; done
