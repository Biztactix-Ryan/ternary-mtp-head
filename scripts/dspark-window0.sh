#!/bin/bash
# Card-0 test window between training runs: the 8-token table kernel (speed and output check), then drafters.
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
export CUDA_VISIBLE_DEVICES=0
{ echo "== table kernel, 5-8 tokens: new"; $B/llama-bench -m out/trunk-tq1p.gguf -ngl 99 -fa 1 -n 0 -p 4,5,6,7,8,9 -r 12 -o md 2>/dev/null | grep "pp" | cut -d"|" -f8-
  echo "== old limit (GGML_CUDA_TQ1P_COLS4=1)"; GGML_CUDA_TQ1P_COLS4=1 $B/llama-bench -m out/trunk-tq1p.gguf -ngl 99 -fa 1 -n 0 -p 4,5,6,7,8,9 -r 12 -o md 2>/dev/null | grep "pp" | cut -d"|" -f8-
  $B/llama-server -m out/trunk-tq1p.gguf -ngl 99 -c 4096 -fa on --host 127.0.0.1 --port 8090 > /dev/null 2>&1 & p1=$!
  GGML_CUDA_TQ1P_COLS4=1 $B/llama-server -m out/trunk-tq1p.gguf -ngl 99 -c 4096 -fa on --host 127.0.0.1 --port 8091 > /dev/null 2>&1 & p2=$!
  for p in 8090 8091; do for i in $(seq 100); do curl -sf http://127.0.0.1:$p/health >/dev/null && break; sleep 2; done; done
  echo "== new kernel vs old path"; python3 mtp/compare_small.py 8091 8090; echo "== old path vs itself"; python3 mtp/compare_small.py 8091 8091
  kill $p1 $p2; wait $p1 $p2 2>/dev/null; } > logs/window0.txt 2>&1
for b in 7 16; do $PY mtp/dspark.py export --ckpt out/ds-F-float/best.pt --out out/dspark-F-b$b.gguf --block $b --io-from $Q4 2>&1 | tail -1 >> logs/window0.txt; done
echo "-- window $(date '+%T')" >> logs/dst.log
bash mtp/dspark-test.sh 0 8097 aligned-b7-n7-cols8 out/dspark-aligned-b7.gguf 7
bash mtp/dspark-test.sh 0 8097 F-b7-n7 out/dspark-F-b7.gguf 7
bash mtp/dspark-test.sh 0 8097 F-b16-n15 out/dspark-F-b16.gguf 15
touch logs/gpu0.go
echo WINDOW0_DONE >> logs/window0.txt
