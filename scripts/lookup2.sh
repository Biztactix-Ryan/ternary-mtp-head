#!/bin/bash
# Baselines for lookup.sh that do not use the MTP head, on the plain trunk file. (The trunk file with an
# in-file head only loads when --spec-type includes draft-mtp; otherwise the loader rejects the extra tensors.)
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
M=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
run() { # gpu port label extra...
  gpu=$1; port=$2; label=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $M -ngl 99 -c 16384 -fa on --host 127.0.0.1 --port $port "$@" > logs/lk-server-$label.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do
    curl -sf http://127.0.0.1:$port/health > /dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then
    python3 mtp/bench12.py $port $label >> logs/lk2-$gpu.log 2>&1
    python3 mtp/bench_copy.py $port $label >> logs/lk2-$gpu.log 2>&1
  else echo "$label: server failed to start: $(grep -iE 'error|invalid|unknown|abort' logs/lk-server-$label.log | tail -2)" >> logs/lk2-$gpu.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
rm -f logs/lk2-0.log logs/lk2-1.log
( run 0 8097 plain
  run 0 8097 ngram-m3  --spec-type ngram-simple --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 3
  run 0 8097 ngram-m48 --spec-type ngram-simple --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 48 ) &
( run 1 8098 ngram-mod --spec-type ngram-mod
  run 1 8098 ngram-m7  --spec-type ngram-simple --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 7 ) &
wait
{ echo "-- baselines without the head (plain trunk file)"; grep -h "TOTAL\|failed" logs/lk2-0.log logs/lk2-1.log; } >> logs/lookup-summary.txt
echo LOOKUP2_DONE >> logs/lookup-summary.txt
