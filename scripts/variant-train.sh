#!/bin/bash
# Capacity variants of the ternary head, same training as the head to head (same init, data, 300 steps).
# usage: variant-train.sh GPU NAME [train_mtp.py args...]   e.g. variant-train.sh 1 r4096 --rank 4096 --rank-kv 1024
export CUDA_VISIBLE_DEVICES=$1; NAME=$2; shift 2
cd ~/ttf
PY=ttf-home/.venv/bin/python
MTP=ttf-home/models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
TTF=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
$PY mtp/train_mtp.py --mtp $MTP --dump data/dump-ttf --emb-from $TTF --init out/s1-infile-float/best.pt \
    --steps 300 --warmup 30 --lr 1e-4 --out out/var-$NAME --ttf "$@" > logs/var-$NAME.log 2>&1
rm -f out/var-$NAME/last.pt
echo VARIANT_DONE >> logs/var-$NAME.log
