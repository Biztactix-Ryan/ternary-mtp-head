#!/bin/bash
cd ~/ttf
PY=ttf-home/.venv/bin/python; Q4=ttf-home/models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf
echo "-- io-final $(date +%H:%M:%S)" >> logs/dst.log
run() { # gpu port tag ckpt ternio
  CUDA_VISIBLE_DEVICES=$1 $PY mtp/dspark.py export --ckpt $4 --out out/dspark-$3-b7.gguf --block 7 --io-from $Q4 --tern-io $5 > logs/export-$3.log 2>&1; tail -1 logs/export-$3.log >> logs/io-final.txt
  bash mtp/dspark-test.sh $1 $2 $3-b7-n5 out/dspark-$3-b7.gguf 5
  bash mtp/dspark-test.sh $1 $2 $3-b7-n7 out/dspark-$3-b7.gguf 7
  bash mtp/dspark-test.sh $1 $2 $3-b7-n5-ngram out/dspark-$3-b7.gguf 5 --spec-type draft-dspark,ngram-mod
}
run 0 8131 I out/ds-I-tern-io/best.pt both &
run 1 8132 J out/ds-J-tern-out/best.pt out &
wait
echo IOFINAL_DONE >> logs/io-final.txt
