#!/bin/bash
# Head to head: ternary vs float in-file head, identical training (same init, data, steps, schedule).
# usage: h2h-train.sh [gpu]
export CUDA_VISIBLE_DEVICES=${1:-1}
cd ~/ttf
PY=ttf-home/.venv/bin/python
MTP=ttf-home/models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
TTF=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
COMMON="--mtp $MTP --dump data/dump-ttf --emb-from $TTF --init out/s1-infile-float/best.pt --steps 300 --warmup 30 --lr 1e-4"
$PY mtp/train_mtp.py $COMMON --out out/h2h-ttf-r1024 --ttf --rank 1024 > logs/h2h-ttf-r1024.log 2>&1
$PY mtp/train_mtp.py $COMMON --out out/h2h-float > logs/h2h-float.log 2>&1
echo H2H_TRAIN_DONE >> logs/h2h-float.log
