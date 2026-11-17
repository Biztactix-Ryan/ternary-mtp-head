#!/bin/bash
# Engine test of a DSpark drafter on the re-packed ternary trunk.
# usage: dspark-test.sh GPU PORT LABEL DRAFT.gguf NMAX [extra server args...]   -> appends to logs/dst.log
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
gpu=$1; port=$2; label=$3; draft=$4; nmax=$5; shift 5
CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m out/trunk-tq1p.gguf -md $draft -ngld 99 -ngl 99 -c 16384 -fa on --host 127.0.0.1 --port $port \
  --spec-type draft-dspark --spec-draft-n-max $nmax "$@" > logs/dst-server-$label.log 2>&1 &
pid=$!; ok=0
for i in $(seq 150); do curl -sf http://127.0.0.1:$port/health > /dev/null && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 2; done
if [ $ok = 1 ]; then
  python3 mtp/bench12.py $port $label 2>&1 | tail -1 >> logs/dst.log
  python3 mtp/bench_copy.py $port $label 2>&1 | tail -1 >> logs/dst.log
else echo "$label: server failed to start" >> logs/dst.log; grep -i "error\|abort\|assert" logs/dst-server-$label.log | tail -4 >> logs/dst.log; fi
kill $pid 2>/dev/null; wait $pid 2>/dev/null
