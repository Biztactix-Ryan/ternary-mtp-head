#!/bin/bash
# card 0 after run F: wait for the engine-test window to close (logs/gpu0.go), then H: 6-layer ternary from F.
cd ~/ttf
PY=ttf-home/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T="$PY mtp/dspark.py train --dump data/dump-ds --trunk ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf"
while ! grep -q TRAIN_DONE logs/ds-F.log 2>/dev/null || [ ! -f logs/gpu0.go ]; do sleep 10; done
HRS=$(python3 -c "import time,calendar; t=time.gmtime(); print(max(0.5, round(17.4 - (t.tm_hour + t.tm_min/60), 2)))")
CUDA_VISIBLE_DEVICES=0 $T --out out/ds-H-tern-6l --init out/ds-F-float/best.pt --tern --layers 6 --anchors 64 --hours $HRS --lr 2e-4 > logs/ds-H.log 2>&1
if ! grep -q TRAIN_DONE logs/ds-H.log && [ $(grep -c "^step" logs/ds-H.log) -lt 20 ]; then
  HRS=$(python3 -c "import time; t=time.gmtime(); print(max(0.5, round(17.4 - (t.tm_hour + t.tm_min/60), 2)))")
  CUDA_VISIBLE_DEVICES=0 $T --out out/ds-H-tern-6l --init out/ds-F-float/best.pt --tern --layers 6 --anchors 40 --max-len 2048 --hours $HRS --lr 2e-4 > logs/ds-H2.log 2>&1
fi
echo DSTRAIN3_DONE > logs/dstrain3.done
