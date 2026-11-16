#!/bin/bash
# Section 7: validate the PyTorch replica with stock weights on the control model, then on the ternary model.
# usage: validate.sh INPUT.jsonl [gpu]   (needs the card free: about 20 GB for the control dump)
IN=${1:-data/gen/conversations.jsonl}
export CUDA_VISIBLE_DEVICES=${2:-1}
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
MTP=ttf-home/models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
mkdir -p data/val/q4 data/val/ttf
rm -f data/val/q4/* data/val/ttf/*
$B/llama-dump-nextn -m ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf -i $IN -o data/val/q4 2>&1 | grep -E "^done|fail|error"
$B/llama-dump-nextn -m ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf -i $IN -o data/val/ttf 2>&1 | grep -E "^done|fail|error"
echo "== stock head replica on control (Q4_K_XL) dump"
$PY mtp/mtp_head.py eval --mtp $MTP --dump data/val/q4 2>&1 | tail -4
echo "== stock head replica on ternary dump"
$PY mtp/mtp_head.py eval --mtp $MTP --dump data/val/ttf 2>&1 | tail -4
