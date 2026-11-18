#!/bin/bash
# Does a wider ternary drafter help? Both runs continue the ternary drafter G for 3 h under the same settings.
#   W  feed-forward width 17408 -> 21760 (1.25x, +0.33B weights)     C  control, G unchanged
cd ~/ttf
PY=ttf-home/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
G=out/ds-G-tern/best.pt
T="$PY mtp/dspark.py train --dump bulk/dump-ds --trunk $Q4 --init $G --tern --anchors 64 --hours ${HRS:-3} --lr 1e-4"
( CUDA_VISIBLE_DEVICES=0 $T --ff 21760 --out out/ds-W-wide > logs/ds-W.log 2>&1; rm -f out/ds-W-wide/last.pt ) &
( CUDA_VISIBLE_DEVICES=1 $T --out out/ds-C-cont > logs/ds-C.log 2>&1; rm -f out/ds-C-cont/last.pt ) &
wait
echo WIDE_DONE > logs/wide.done
