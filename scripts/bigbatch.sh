#!/bin/bash
# After the large-batch engine fix (native TQ1_P MMQ tiles): parallel slots and long drafts again.
# Results: logs/bigbatch-summary.txt  (before the fix: logs/parallel-summary.txt, logs/deep-summary.txt)
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
RP=out/trunk-tq1p.gguf
RPH=out/ttf-27b-mtp-tq1p.gguf
rm -f logs/bb-0.log logs/bb-1.log
serve() { # gpu port np model extra...  -> PID
  gpu=$1; port=$2; np=$3; model=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $model -ngl 99 -c $((8192 * np)) -np $np -fa on --host 127.0.0.1 --port $port "$@" > logs/bb-server-$port.log 2>&1 &
  PID=$!
  for i in $(seq 150); do curl -sf http://127.0.0.1:$port/health > /dev/null && return 0; kill -0 $PID 2>/dev/null || return 1; sleep 2; done
  return 1
}
par() { # gpu port label np model extra...
  gpu=$1; port=$2; label=$3; np=$4; model=$5; shift 5
  if serve $gpu $port $np $model "$@"; then $PY mtp/bench_parallel.py $port $label $np >> logs/bb-$gpu.log 2>&1; else echo "$label np=$np: server failed to start" >> logs/bb-$gpu.log; fi
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
}
single() { # gpu port label model extra...
  gpu=$1; port=$2; label=$3; model=$4; shift 4
  if serve $gpu $port 1 $model "$@"; then
    python3 mtp/bench12.py $port $label 2>&1 | tail -1 >> logs/bb-$gpu.log
    python3 mtp/bench_copy.py $port $label 2>&1 | tail -1 >> logs/bb-$gpu.log
  else echo "$label: server failed to start" >> logs/bb-$gpu.log; fi
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
}
H="--spec-type draft-mtp"
HL="--spec-type draft-mtp,ngram-mod"
( for np in 1 2 4 8; do par 0 8097 rp-plain $np $RP; done
  for np in 2 4 8; do par 0 8097 rp-head3 $np $RPH $H --spec-draft-n-max 3; done
  single 0 8097 plain $RP
  single 0 8097 lookup-n16 $RP --spec-type ngram-mod --spec-draft-n-max 16 ) &
( for np in 2 4 8; do par 1 8098 rp-head1 $np $RPH $H --spec-draft-n-max 1; done
  single 1 8098 head-n3 $RPH $H --spec-draft-n-max 3
  single 1 8098 head-n8 $RPH $H --spec-draft-n-max 8
  single 1 8098 head+lookup-n2 $RPH $HL --spec-draft-n-max 2
  single 1 8098 head+lookup-n16 $RPH $HL --spec-draft-n-max 16
  single 1 8098 head+lookup-n16-p0.6 $RPH $HL --spec-draft-n-max 16 --spec-draft-p-min 0.6 ) &
wait
{ echo "== after the large-batch fix $(date '+%F %T')"; cat logs/bb-0.log logs/bb-1.log; } > logs/bigbatch-summary.txt
echo BIGBATCH_DONE >> logs/bigbatch-summary.txt
