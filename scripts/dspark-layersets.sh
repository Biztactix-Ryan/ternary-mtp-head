#!/bin/bash
# Which trunk layers should the drafter read? Float drafters, each 1.3 h from the fine-tuned float drafter F, whose
# first layer is refitted to the new layer set by regression (dspark_refit.py). Ten-layer dump columns:
#   0:6 1:13 2:20 3:27 4:34 5:41 6:48 7:55 8:62 9:63
#   S0 stock five (control, F continued)   S1 late five 34,48,55,62,63   S2 all ten   S3 three 20,41,62
cd ~/ttf
PY=ttf-home/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
DUMP=bulk/dump-ds10
F=out/ds-F-float/best.pt
while [ ! -f logs/dsdump10.done ]; do sleep 20; done
train() { # gpu name init
  CUDA_VISIBLE_DEVICES=$1 $PY mtp/dspark.py train --dump $DUMP --dump-layers 10 --trunk $Q4 --init $3 --out out/ds-$2 --hours 1.3 --lr 5e-5 > logs/ds-$2.log 2>&1
  rm -f out/ds-$2/last.pt
}
refit() { # gpu name cols
  CUDA_VISIBLE_DEVICES=$1 $PY mtp/dspark_refit.py $DUMP $F $3 out/refit-$2.pt > logs/refit-$2.log 2>&1
}
( refit 0 S1 4,6,7,8,9 && train 0 S1 out/refit-S1.pt; train 0 S0 $F ) &
( refit 1 S2 0,1,2,3,4,5,6,7,8,9 && train 1 S2 out/refit-S2.pt; refit 1 S3 2,5,8 && train 1 S3 out/refit-S3.pt ) &
wait
echo LAYERSETS_DONE > logs/layersets.done
