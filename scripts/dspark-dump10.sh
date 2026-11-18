#!/bin/bash
# Trunk features at ten depths (inputs of layers 6,13,...,62 and 63) for the layer-set experiments; the same 2,400
# sequences as data/dump-ds, whose five layers are columns 0,2,4,6,8 here. Results: bulk/dump-ds10 (about 245 GB)
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
mkdir -p bulk/dump-ds10
L=6,13,20,27,34,41,48,55,62,63
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $B/llama-dump-nextn -m out/trunk-tq1p.gguf -i data/prep3/in.$g.jsonl -o bulk/dump-ds10 --prefix pub$g -c 8192 -b 512 --layers $L --shard-positions 50000 > logs/dsdump10-$g.log 2>&1 &
done
wait
echo DSDUMP10_DONE > logs/dsdump10.done
