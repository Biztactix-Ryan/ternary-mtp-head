#!/bin/bash
# Stock DSpark drafter (RadixArk/Qwen3.8-27B-DSpark, block 7) on the ternary model (floor) and on the Q4_K_XL
# control (ceiling for a retrain), with cycle timers. Results: logs/dspark-base-summary.txt
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
RP=out/trunk-tq1p.gguf
Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
D=ttf-home/models/drafts/dspark-27b/gguf/Qwen3.8-27B-DSpark-bf16.gguf
while [ ! -f ttf-home/models/drafts/dspark-27b/download.done ]; do sleep 10; done
rm -f logs/dsb-0.log logs/dsb-1.log
run() { # gpu port label model extra...
  gpu=$1; port=$2; label=$3; model=$4; shift 4
  TTF_SPEC_TIMING=1 CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $model -ngl 99 -c 16384 -fa on --host 127.0.0.1 --port $port "$@" > logs/dsb-server-$label.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do curl -sf http://127.0.0.1:$port/health > /dev/null && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 2; done
  if [ $ok = 1 ]; then
    python3 mtp/bench12.py $port $label 2>&1 | tail -1 >> logs/dsb-$gpu.log
    python3 mtp/bench_copy.py $port $label 2>&1 | tail -1 >> logs/dsb-$gpu.log
  else echo "$label: server failed to start" >> logs/dsb-$gpu.log; tail -5 logs/dsb-server-$label.log >> logs/dsb-$gpu.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  grep -h "spec timing\|TTF_SPEC" logs/dsb-server-$label.log | tail -3 >> logs/dsb-$gpu.log
}
DS="-md $D -ngld 99 --spec-type draft-dspark"
( run 0 8097 ttf-dspark7 $RP $DS --spec-draft-n-max 7
  run 0 8097 ttf-dspark3 $RP $DS --spec-draft-n-max 3 ) &
( run 1 8098 q4-plain $Q4
  run 1 8098 q4-dspark7 $Q4 $DS --spec-draft-n-max 7
  run 1 8098 q4-dspark3 $Q4 $DS --spec-draft-n-max 3
  run 1 8098 q4-dspark7-p0.5 $Q4 $DS --spec-draft-n-max 7 --spec-draft-p-min 0.5 ) &
wait
{ echo "== stock DSpark $(date '+%F %T')"; cat logs/dsb-0.log logs/dsb-1.log; } > logs/dspark-base-summary.txt
echo DSPARK_BASE_DONE >> logs/dspark-base-summary.txt
