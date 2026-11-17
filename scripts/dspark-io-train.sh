#!/bin/bash
cd ~/ttf
PY=ttf-home/.venv/bin/python; Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
CUDA_VISIBLE_DEVICES=0 $PY mtp/dspark.py train --dump data/dump-ds --trunk $Q4 --init out/ds-G-tern/best.pt --tern --tern-io both --out out/ds-I-tern-io --hours 1.05 --lr 1e-4 > logs/ds-I.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY mtp/dspark.py train --dump data/dump-ds --trunk $Q4 --init out/ds-G-tern/best.pt --tern --tern-io out --out out/ds-J-tern-out --hours 1.05 --lr 1e-4 > logs/ds-J.log 2>&1 &
wait
