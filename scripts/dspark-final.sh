#!/bin/bash
# Export the ternary drafters and test them in the engine: G on card 0, H on card 1.
cd ~/ttf
PY=$(grep -o '^PY=.*' mtp/window0.sh | head -1 | cut -d= -f2-); PY=${PY:-python3}
Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
echo "-- final $(date +%H:%M:%S)" >> logs/dst.log
run() { # gpu port tag ckpt
  for b in 7 16; do CUDA_VISIBLE_DEVICES=$1 $PY mtp/dspark.py export --ckpt $4 --out out/dspark-$3-b$b.gguf --block $b --io-from $Q4 2>&1 | tail -1 >> logs/final.txt; done
  bash mtp/dspark-test.sh $1 $2 $3-b7-n7 out/dspark-$3-b7.gguf 7
  bash mtp/dspark-test.sh $1 $2 $3-b16-n15 out/dspark-$3-b16.gguf 15
  bash mtp/dspark-test.sh $1 $2 $3-b7-n7-ngram out/dspark-$3-b7.gguf 7 --spec-type draft-dspark,ngram-mod
  bash mtp/dspark-test.sh $1 $2 $3-b7-n5 out/dspark-$3-b7.gguf 5
}
run 0 8131 G out/ds-G-tern/best.pt &
run 1 8132 H out/ds-H-tern-6l/best.pt &
wait
ls -la out/dspark-*.gguf >> logs/final.txt
echo FINAL_DONE >> logs/final.txt
