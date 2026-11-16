#!/bin/bash
# Prepare the generation log and dump ternary hidden states, half per card.
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
M=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
mkdir -p data/dump-ttf data/prep
rm -f data/dump-ttf/*
$PY mtp/prep_dump.py data/gen/conversations.jsonl data/prep/in 2
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $B/llama-dump-nextn -m $M -i data/prep/in.$g.jsonl -o data/dump-ttf --prefix gpu$g \
    > logs/dump-$g.log 2>&1 &
done
wait
tail -1 logs/dump-0.log logs/dump-1.log
du -sh data/dump-ttf
