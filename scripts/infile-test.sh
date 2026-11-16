#!/bin/bash
# Engine test of a trunk GGUF with an in-file MTP head (no -md).
# usage: infile-test.sh MODEL.gguf LABEL [gpu] [draft counts...]
MODEL=$1; LABEL=$2; export CUDA_VISIBLE_DEVICES=${3:-1}; shift 3
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PORT=8097
for n in ${@:-1 2 3}; do
  $B/llama-server -m $MODEL -ngl 99 -c 8192 -fa on --host 127.0.0.1 --port $PORT \
     --spec-type draft-mtp --spec-draft-n-max $n > logs/server-$LABEL-n$n.log 2>&1 &
  pid=$!
  ok=0
  for i in $(seq 150); do
    curl -sf http://127.0.0.1:$PORT/health > /dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then python3 mtp/bench3.py $PORT $LABEL-n$n; else echo "server failed to start:"; grep -iE "error|fail|assert|abort" logs/server-$LABEL-n$n.log | tail -8; tail -3 logs/server-$LABEL-n$n.log; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  [ $ok = 1 ] || exit 1
done
