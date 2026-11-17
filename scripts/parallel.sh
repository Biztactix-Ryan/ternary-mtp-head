#!/bin/bash
# Parallel slots: total throughput with 1, 2, 4 and 8 concurrent requests, original vs re-packed, and the
# head at 2 and 4 slots. Starts after the accuracy run. Results: logs/parallel-summary.txt
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
ORIG=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
RP=out/trunk-tq1p.gguf
RPH=out/ttf-27b-mtp-tq1p.gguf
while ! grep -q ACCURACY_DONE logs/accuracy-summary.txt 2>/dev/null; do sleep 30; done
rm -f logs/par-0.log logs/par-1.log
par() { # gpu port label np model extra...
  gpu=$1; port=$2; label=$3; np=$4; model=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $model -ngl 99 -c $((8192 * np)) -np $np -fa on --host 127.0.0.1 --port $port "$@" > logs/par-server-$port.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do curl -sf http://127.0.0.1:$port/health > /dev/null && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 2; done
  if [ $ok = 1 ]; then $PY mtp/bench_parallel.py $port $label $np >> logs/par-$gpu.log 2>&1; else echo "$label np=$np: server failed to start" >> logs/par-$gpu.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
( for np in 1 2 4 8; do par 0 8097 orig-plain $np $ORIG; done
  for np in 2 4; do par 0 8097 rp-head1 $np $RPH --spec-type draft-mtp --spec-draft-n-max 1; done ) &
( for np in 1 2 4 8; do par 1 8098 rp-plain $np $RP; done
  for np in 2 4; do par 1 8098 rp-head3 $np $RPH --spec-type draft-mtp --spec-draft-n-max 3; done ) &
wait
{ echo "== parallel slots $(date '+%F %T')"; sort logs/par-0.log logs/par-1.log; } > logs/parallel-summary.txt
echo PARALLEL_DONE >> logs/parallel-summary.txt
