#!/bin/bash
# Lookup (n-gram) drafting on the ternary model, alone and combined with the in-file MTP head.
# Two prompt sets: the 12 general prompts (bench12.py) and 6 copy-heavy edits (bench_copy.py).
# Draft length matters on this engine: a verify call of up to 4 tokens uses the fused ternary kernel,
# 5-8 the generic one, and more than 8 costs a flat ~258 ms. Results: logs/lookup-summary.txt
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
M=out/nx-ttf4096-s2long.gguf          # ternary trunk + in-file ternary rank-4096 head
while ! grep -q SCALE_DONE logs/scale-0.5.log 2>/dev/null; do sleep 30; done

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
    python3 mtp/bench12.py $port $label >> logs/lk-$gpu.log 2>&1
    python3 mtp/bench_copy.py $port $label >> logs/lk-$gpu.log 2>&1
  else echo "$label: server failed to start: $(grep -iE 'error|invalid|unknown|abort' logs/lk-server-$label.log | tail -2)" >> logs/lk-$gpu.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
rm -f logs/lk-0.log logs/lk-1.log
(
  run 0 8097 plain
  run 0 8097 mtp1            --spec-type draft-mtp --spec-draft-n-max 1
  run 0 8097 mtp2            --spec-type draft-mtp --spec-draft-n-max 2
  run 0 8097 ngram-m3        --spec-type ngram-simple --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 3
  run 0 8097 ngram-m7        --spec-type ngram-simple --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 7
  run 0 8097 ngram-m48       --spec-type ngram-simple --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 48
) &
(
  run 1 8098 ngram-mod       --spec-type ngram-mod
  run 1 8098 mtp1+ngram-m3   --spec-type draft-mtp,ngram-simple --spec-draft-n-max 1 --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 3
  run 1 8098 mtp1+ngram-m48  --spec-type draft-mtp,ngram-simple --spec-draft-n-max 1 --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 48
  run 1 8098 mtp1+ngram-mod  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 1
  run 1 8098 ngram-m3+mtp1   --spec-type ngram-simple,draft-mtp --spec-draft-n-max 1 --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 3
  run 1 8098 ngram-m48+mtp1  --spec-type ngram-simple,draft-mtp --spec-draft-n-max 1 --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 48
) &
wait
{ echo "== lookup drafting summary $(date '+%F %T')"; echo "-- general prompts (12)"; grep -h " TOTAL" logs/lk-0.log logs/lk-1.log
  echo "-- copy-heavy edits (6)"; grep -h "COPY-TOTAL" logs/lk-0.log logs/lk-1.log; grep -h "failed" logs/lk-0.log logs/lk-1.log; } > logs/lookup-summary.txt
echo LOOKUP_DONE >> logs/lookup-summary.txt
