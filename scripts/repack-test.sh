#!/bin/bash
# Engine tests of the re-packed (all TQ1_P) trunk with the in-file ternary head, also re-packed.
# Results: logs/repack-summary.txt
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
T=out/trunk-tq1p.gguf
$PY mtp/build_infile.py $T out/tmp-combined.gguf --ttf-ckpt out/nx-ttf4096-s2long/best.pt > logs/repack-build.log 2>&1
CUDA_VISIBLE_DEVICES=1 $PY mtp/repack_tq1p.py out/tmp-combined.gguf out/ttf-27b-mtp-tq1p.gguf >> logs/repack-build.log 2>&1
rm -f out/tmp-combined.gguf
M=out/ttf-27b-mtp-tq1p.gguf
run() { # gpu port label model extra...
  gpu=$1; port=$2; label=$3; model=$4; shift 4
  TTF_SPEC_TIMING=1 CUDA_VISIBLE_DEVICES=$gpu $B/llama-server -m $model -ngl 99 -c 16384 -fa on --host 127.0.0.1 --port $port "$@" > logs/rp-server-$label.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 150); do
    curl -sf http://127.0.0.1:$port/health > /dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then
    python3 mtp/bench12.py $port $label >> logs/rp-$gpu.log 2>&1
    python3 mtp/bench_copy.py $port $label >> logs/rp-$gpu.log 2>&1
  else echo "$label: server failed to start: $(grep -iE 'error|invalid|expected|abort' logs/rp-server-$label.log | tail -2)" >> logs/rp-$gpu.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  grep -h spc_timing logs/rp-server-$label.log | sed "s/^/$label /" >> logs/rp-$gpu.log
}
rm -f logs/rp-0.log logs/rp-1.log
( run 0 8097 plain $T
  run 0 8097 mtp1 $M --spec-type draft-mtp --spec-draft-n-max 1
  run 0 8097 mtp3 $M --spec-type draft-mtp --spec-draft-n-max 3 ) &
( run 1 8098 mtp2 $M --spec-type draft-mtp --spec-draft-n-max 2
  run 1 8098 mtp2+ngram-mod $M --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2
  run 1 8098 ngram-mod $T --spec-type ngram-mod ) &
wait
{ echo "== re-packed trunk + head $(date '+%F %T')"; tail -2 logs/repack-build.log
  ls -l $T $M | awk '{printf "%.0f MiB  %s\n", $5/1048576, $9}'
  echo "-- general prompts (12)"; grep -h " TOTAL" logs/rp-0.log logs/rp-1.log
  echo "-- copy-heavy edits (6)"; grep -h "COPY-TOTAL" logs/rp-0.log logs/rp-1.log
  grep -h "spc_timing\|failed" logs/rp-0.log logs/rp-1.log; } > logs/repack-summary.txt
echo REPACK_DONE >> logs/repack-summary.txt
