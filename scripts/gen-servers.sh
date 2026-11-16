#!/bin/bash
# Start (or stop) the data-generation servers: one ternary llama-server per card, 4 slots of 8192.
# usage: gen-servers.sh start [gpu ...] | stop
cd ~/ttf/ttf-home
B=engine-q38/build-cuda/bin
M=models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
mkdir -p ~/ttf/logs
if [ "$1" = stop ]; then pkill -f "llama-server.*--port 810" ; exit 0; fi
shift
for g in ${@:-0 1}; do
  port=$((8101 + g))
  CUDA_VISIBLE_DEVICES=$g nohup $B/llama-server -m $M -ngl 99 -c 49152 -np 4 -fa on --host 127.0.0.1 --port $port \
    > ~/ttf/logs/gen-$port.log 2>&1 &
done
for g in ${@:-0 1}; do
  port=$((8101 + g))
  for i in $(seq 120); do curl -sf http://127.0.0.1:$port/health > /dev/null && break; sleep 2; done
  echo "port $port: $(curl -s http://127.0.0.1:$port/health)"
done
