#!/bin/bash
# Answer accuracy on 60 general-knowledge and 60 IT-support questions, for each configuration and the
# Q4_K_XL control. Starts after the wider validation run. Results: logs/accuracy-summary.txt
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
ORIG=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
RP=out/trunk-tq1p.gguf
RPH=out/ttf-27b-mtp-tq1p.gguf
Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
while ! grep -q VALIDATE_DONE logs/validate-summary.txt 2>/dev/null; do sleep 30; done
rm -f logs/acc-0.log logs/acc-1.log
run() { # gpu port label model extra...
  gpu=$1; port=$2; label=$3; model=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $model -ngl 99 -c 8192 -fa on --host 127.0.0.1 --port $port "$@" > logs/acc-server-$label.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do curl -sf http://127.0.0.1:$port/health > /dev/null && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 2; done
  if [ $ok = 1 ]; then $PY mtp/qa_accuracy.py $port $label >> logs/acc-$gpu.log 2>&1; else echo "$label: server failed to start" >> logs/acc-$gpu.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
( run 0 8097 orig-plain $ORIG
  run 0 8097 rp-head3 $RPH --spec-type draft-mtp --spec-draft-n-max 3
  run 0 8097 rp-best $RPH --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 ) &
( run 1 8098 rp-plain $RP
  run 1 8098 q4-control $Q4 ) &
wait
{ echo "== answer accuracy $(date '+%F %T')"; grep -hv MISS logs/acc-0.log logs/acc-1.log
  echo; echo "-- do the configurations give the same answers?"
  for b in rp-plain rp-head3 rp-best; do $PY mtp/qa_accuracy.py compare data/validate orig-plain $b; done
  $PY mtp/qa_accuracy.py compare data/validate rp-plain rp-head3
  echo; echo "-- answers graded wrong (keyword match; check by eye)"; grep -h MISS logs/acc-0.log logs/acc-1.log | sort -k3,3 -k2,2; } > logs/accuracy-summary.txt 2>&1
echo ACCURACY_DONE >> logs/accuracy-summary.txt
