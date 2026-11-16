#!/bin/bash
# Unattended follow-on for the bulk run: when both batches (public prompts, scrubbed sessions) finish,
# stop the generation servers and dump hidden states for the new data, half per card. Nothing is trained here.
cd ~/ttf
B=ttf-home/engine-q38/build-cuda/bin
PY=ttf-home/.venv/bin/python
M=ttf-home/models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
while ! (grep -q "^batch done" logs/batch2.log && grep -q "^batch done" logs/batch3.log); do sleep 60; done
bash mtp/gen-servers.sh stop; sleep 5
mkdir -p data/dump-ttf2 data/prep2
$PY mtp/prep_dump.py data/gen2/conversations.jsonl data/prep2/pub 2 > logs/after-batch2.log 2>&1
$PY mtp/prep_dump.py data/gen3/conversations.jsonl data/prep2/sess 2 >> logs/after-batch2.log 2>&1
for g in 0 1; do
  ( for s in pub sess; do
      CUDA_VISIBLE_DEVICES=$g $B/llama-dump-nextn -m $M -c 12288 --max-prompt-rows 1536 -i data/prep2/$s.$g.jsonl -o data/dump-ttf2 --prefix $s$g \
        > logs/dump2-$s-$g.log 2>&1
    done ) &
done
wait
grep -h "^done" logs/dump2-*.log >> logs/after-batch2.log
du -sh data/dump-ttf2 >> logs/after-batch2.log
echo AFTER_BATCH2_DONE >> logs/after-batch2.log
