#!/bin/bash
# Data-scaling ablation: the float head trained exactly as on-float-s1 (1,500 steps from the same start)
# but on 12.5%, 25% and 50% of the training conversations. 100% is the existing on-float-s1 run.
cd ~/ttf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=ttf-home/.venv/bin/python
MTP=ttf-home/models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
TTF=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
run() { # gpu frac
  CUDA_VISIBLE_DEVICES=$1 $PY mtp/train_mtp.py --mtp $MTP --dump data/dump-ttf,data/dump-ttf2 --emb-from $TTF --half-state \
      --eval-seqs 150 --eval-every 100 --warmup 50 --lr 1e-4 --init out/h2h-float/best.pt --steps 1500 \
      --train-frac $2 --out out/scale-$2 > logs/scale-$2.log 2>&1
  rm -f out/scale-$2/best.pt
}
( run 0 0.25; run 0 0.125 ) &
( run 1 0.5 ) &
wait
echo SCALE_DONE >> logs/scale-0.5.log
