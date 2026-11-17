#!/bin/bash
# Trunk features for DSpark training: inputs of trunk layers 6,20,34,48,62 at every position (plus the usual
# final hidden state and top-32). Ternary trunk: 1,200 public-prompt sequences per card. Q4 control: 80
# sequences, to validate the PyTorch replica against the stock drafter. Results: data/dump-ds, data/dump-ds-q4
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
while ! grep -q DSPARK_BASE_DONE logs/dspark-base-summary.txt 2>/dev/null; do sleep 15; done
mkdir -p data/prep3 data/dump-ds data/dump-ds-q4
head -1200 data/prep2/pub.0.jsonl > data/prep3/in.0.jsonl
head -1200 data/prep2/pub.1.jsonl > data/prep3/in.1.jsonl
tail -80 data/prep2/pub.1.jsonl > data/prep3/q4.jsonl
L=6,20,34,48,62
( CUDA_VISIBLE_DEVICES=1 $B/llama-dump-nextn -m ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf -i data/prep3/q4.jsonl -o data/dump-ds-q4 --prefix q4 -c 8192 -b 512 --layers $L --shard-positions 50000 > logs/dsdump-q4.log 2>&1
  echo Q4_DUMP_DONE >> logs/dsdump-q4.log
  CUDA_VISIBLE_DEVICES=1 $B/llama-dump-nextn -m out/trunk-tq1p.gguf -i data/prep3/in.1.jsonl -o data/dump-ds --prefix pub1 -c 8192 -b 512 --layers $L --shard-positions 50000 > logs/dsdump-1.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=0 $B/llama-dump-nextn -m out/trunk-tq1p.gguf -i data/prep3/in.0.jsonl -o data/dump-ds --prefix pub0 -c 8192 -b 512 --layers $L --shard-positions 50000 > logs/dsdump-0.log 2>&1 &
wait
echo DSDUMP_DONE > logs/dsdump.done
