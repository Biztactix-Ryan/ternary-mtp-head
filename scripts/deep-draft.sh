#!/bin/bash
# Deeper drafting on the re-packed model: 4, 5, 6 and 8 drafts per cycle, and confidence-gated deep drafting.
# Results: logs/deep-summary.txt
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
M=out/ttf-27b-mtp-tq1p.gguf
run() { # gpu port label extra...
  gpu=$1; port=$2; label=$3; shift 3
  TTF_SPEC_TIMING=1 CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $M -ngl 99 -c 16384 -fa on --host 127.0.0.1 --port $port --spec-type draft-mtp "$@" > logs/dd-server-$label.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do
    curl -sf http://127.0.0.1:$port/health > /dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then
    python3 mtp/bench12.py $port $label >> logs/dd-$gpu.log 2>&1
    python3 mtp/bench_copy.py $port $label >> logs/dd-$gpu.log 2>&1
  else echo "$label: server failed to start" >> logs/dd-$gpu.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
rm -f logs/dd-0.log logs/dd-1.log
( for n in 4 5 6 8; do run 0 8097 n$n --spec-draft-n-max $n; done ) &
( run 1 8098 n3-pmin0.5 --spec-draft-n-max 3 --spec-draft-p-min 0.5
  run 1 8098 n6-pmin0.5 --spec-draft-n-max 6 --spec-draft-p-min 0.5
  run 1 8098 n6-pmin0.7 --spec-draft-n-max 6 --spec-draft-p-min 0.7
  run 1 8098 n8-pmin0.8 --spec-draft-n-max 8 --spec-draft-p-min 0.8 ) &
wait
{ echo "== deeper drafting, re-packed model $(date '+%F %T')"; echo "-- general prompts (12)"; grep -h " TOTAL" logs/dd-0.log logs/dd-1.log
  echo "-- copy-heavy edits (6)"; grep -h "COPY-TOTAL" logs/dd-0.log logs/dd-1.log; grep -h failed logs/dd-0.log logs/dd-1.log; } > logs/deep-summary.txt
echo DEEP_DONE >> logs/deep-summary.txt
