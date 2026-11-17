#!/bin/bash
# Wider validation of the re-packed model with the in-file ternary head. Results: logs/validate-summary.txt
#   suites: 12 general prompts, 6 copy-heavy edits, 40 held-out coding-agent turns (long history, tools)
#   variants: greedy / temperature 0.7, thinking off / on
#   configurations: original plain, re-packed plain, re-packed + head (3 drafts), re-packed + head + lookup
#   then: parallel slots
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
ORIG=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
RP=out/trunk-tq1p.gguf
RPH=out/ttf-27b-mtp-tq1p.gguf
rm -rf data/validate; mkdir -p data/validate
rm -f logs/val-0.log logs/val-1.log logs/val-par.log

serve() { # gpu port np model extra...   -> sets PID
  gpu=$1; port=$2; np=$3; model=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $model -ngl 99 -c $((16384 * np)) -np $np -fa on --host 127.0.0.1 --port $port "$@" > logs/val-server-$port.log 2>&1 &
  PID=$!
  for i in $(seq 150); do curl -sf http://127.0.0.1:$port/health > /dev/null && return 0; kill -0 $PID 2>/dev/null || return 1; sleep 2; done
}
suite() { # gpu port label model extra...
  gpu=$1; port=$2; label=$3; model=$4; shift 4
  serve $gpu $port 1 $model "$@" || { echo "$label: server failed to start" >> logs/val-$gpu.log; return; }
  V="$PY mtp/validate.py $port $label"
  { $V general; $V general --think --max-tokens 600; $V general --temp 0.7; $V general --temp 0.7 --think --max-tokens 600
    $V copy --max-tokens 900; $V copy --temp 0.7 --max-tokens 900
    $V sessions --n 40 --max-tokens 400; $V sessions --n 40 --max-tokens 400 --temp 0.7; } >> logs/val-$gpu.log 2>&1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
}
( suite 0 8097 orig-plain $ORIG
  suite 0 8097 rp-head3   $RPH --spec-type draft-mtp --spec-draft-n-max 3 ) &
( suite 1 8098 rp-plain   $RP
  suite 1 8098 rp-best    $RPH --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 ) &
wait

# parallel slots (one card at a time so the load generator does not interfere)
par() { # label np model extra...
  label=$1; np=$2; model=$3; shift 3
  serve 0 8097 $np $model "$@" || { echo "$label np=$np: server failed to start" >> logs/val-par.log; return; }
  python3 mtp/bench_parallel.py 8097 $label $np >> logs/val-par.log 2>&1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
}
for np in 1 2 4 8; do par orig-plain $np $ORIG; par rp-plain $np $RP; done
for np in 2 4; do par rp-head1 $np $RPH --spec-type draft-mtp --spec-draft-n-max 1; par rp-head3 $np $RPH --spec-type draft-mtp --spec-draft-n-max 3; done

{ echo "== wider validation $(date '+%F %T')"
  echo "-- per suite and variant"; sort -k2,2 -k1,1 logs/val-0.log logs/val-1.log | grep -v ": error"
  echo; echo "-- request errors"; grep -c ": error" logs/val-0.log logs/val-1.log
  echo; echo "-- same text as plain decoding? (greedy runs)"
  $PY mtp/validate.py compare data/validate rp-plain rp-head3 | grep "t0 "
  $PY mtp/validate.py compare data/validate rp-plain rp-best | grep "t0 "
  $PY mtp/validate.py compare data/validate orig-plain rp-plain | grep "t0 "
  echo; echo "-- parallel slots"; cat logs/val-par.log; } > logs/validate-summary.txt 2>&1
echo VALIDATE_DONE >> logs/validate-summary.txt
