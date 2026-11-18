#!/bin/bash
# Train the ternary output head itself (layers of the fully ternary drafter I frozen), two learning rates.
cd ~/ttf
PY=ttf-home/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
T="$PY mtp/dspark.py train --dump bulk/dump-ds --trunk $Q4 --init out/ds-I-tern-io/best.pt --tern --tern-io both --train-head --hours ${HRS:-2}"
CUDA_VISIBLE_DEVICES=0 $T --lr ${LR0:-1e-5} --out out/ds-K1-head ${EXTRA} > logs/ds-K1.log 2>&1 &
[ -n "$ONE" ] || CUDA_VISIBLE_DEVICES=1 $T --lr 5e-5 --out out/ds-K2-head > logs/ds-K2.log 2>&1 &
wait
echo HEAD_DONE > logs/head.done
