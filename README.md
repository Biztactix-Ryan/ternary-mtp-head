# A ternary, in-file MTP draft head for the ternary Qwen3.8-27B

> **Credit.** The ternary Qwen3.8-27B conversion, the TTF format and the llama.cpp patches that run it
> are the work of [`oguzeldereli`](https://github.com/oguzeldereli) (code at
> [`oguzeldereli/ttf-home`](https://github.com/oguzeldereli/ttf-home), model at
> `oguzeldereli/ttf-Qwen3.8-27B-GGUF`). This repository only adds a draft head on top of that
> work; none of the model conversion is ours.

Speculative decoding with an MTP (multi-token prediction) draft head does not work out of the box on
the ternary (TTF, 1.6 bits per weight) Qwen3.8-27B: the stock head was trained against the
full-precision model and almost never agrees with the ternary one. This repository holds the code,
the engine patch and the measurements from two days of fixing that:

1. fine-tune the head on hidden states dumped from the ternary model itself;
2. move the head inside the ternary GGUF as block 64, so it shares the trunk's embedding and output matrices;
3. make the head itself ternary, in the same TTF format the trunk uses, trained quantisation-aware;
4. train it for multi-token drafting on its own outputs.

Result: draft acceptance at one draft went from 2–10% (stock head) to about 75%, with a head that adds
181 MiB to the model file instead of 810 MiB. Together with re-packing the model's older-layout weights
(section 8), decoding goes from 35.7 to about 55 tok/s (1.5x) on general prompts and to about 84 tok/s
(2.4x) on copy-heavy edits, on 2x RTX 5000 Ada. The re-pack alone makes plain decoding 11% faster.

Later additions: an engine fix for large batches (section 11) and a fully ternary DSpark parallel
drafter, 60.6 tok/s on general prompts and 176 tok/s on edits with a 1.0 GB draft file (section 12).

No model weights and no datasets are included. Everything here is code, a small engine patch, the
hand-written prompt tasks used for the first smoke test, and result logs.

## Setup these numbers come from

- Target: `ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf` from `oguzeldereli/ttf-Qwen3.8-27B-GGUF`.
- Stock head: `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` from `unsloth/Qwen3.8-27B-GGUF`.
- Engine: llama.cpp at `ca89968` plus the 12 TTF patches from `oguzeldereli/ttf-home` (`ttf engine build --q38`),
  plus the patch in `engine/`.
- Hardware: 2x RTX 5000 Ada (32 GB), one card per process.
- Engine numbers: greedy decoding, thinking off, 8k context, 350 new tokens per prompt, either the
  3 prompts in `scripts/bench3.py` (day 1) or the 12 in `scripts/bench12.py` (day 2).
- "Agreement" is measured in PyTorch on held-out conversations: how often the head's top token equals
  the trunk's top token at positions the model generated. "Acceptance" is the engine's own count of
  accepted drafts.

## Results

### 1. The stock head on the ternary model (3 prompts)

| Target | Drafting | tok/s | Drafts accepted |
| :--- | :--- | ---: | ---: |
| Ternary 27B | none | 35.6–35.7 | n/a |
| Ternary 27B | stock MTP, 3 drafts | 18.0–19.2 | 1.8–4.1% |
| Ternary 27B | stock MTP, 2 drafts | 21.3–23.0 | 2.2–6.6% |
| Q4_K_XL (control) | stock MTP, 3 drafts | 59.4–75.6 | 61.4–88.3% |

A PyTorch replica of the head with stock weights agrees with the control model on 93% of generated
positions and with the ternary model on 9% (small sample), so the replica is right and the problem
is the head, not the method.

### 2. Smoke test: fine-tune the stock head on the ternary model's own hidden states

336k generated tokens from 310 conversations, 250 steps (about 4 minutes).

| Variant | Trainable parameters | Held-out agreement before | After |
| :--- | ---: | ---: | ---: |
| Whole head | 424.7M | 4.6% | 59.9% |
| Input projection and two input norms only | 52.4M | 4.6% | 53.2% |

In the engine at 1 draft the fine-tuned head accepted 38–78% of drafts (stock: 6–10%).
The held-out set for this stage was small (8 conversations).

### 3. Ternary versus float head, identical training (same start, data and 300 steps)

| Head | Trits | Size added | Held-out agreement |
| :--- | ---: | ---: | ---: |
| Float F16 | n/a | 810 MiB | 61.9% |
| Ternary, rank-1024 adapters | 542M | 108 MiB | 57.1% |
| Ternary, rank-4096 adapters | 894M | 171 MiB | 61.3% |
| Ternary, 2x-wide FFN | 863M | 165 MiB | 57.8% |
| Ternary, 2 stacked blocks | 1,016M | 194 MiB | 57.2% |

More ternary parameters close the gap to float when they go into the low-rank adapters. Extra width
and depth did not help at this training length (and would need engine changes).

### 4. Ten times the data (day 2)

3.2M generated tokens: 2.86M from public prompt sets and 341k from scrubbed coding-agent sessions.
6,174 training sequences; held-out 150 sequences from 97 conversations. This held-out set is harder
than the first one, so percentages are not comparable with sections 2 and 3.

| Head | Size added | Steps | First-draft agreement |
| :--- | ---: | ---: | ---: |
| Float F16 | 810 MiB | 1,500 | 70.3% |
| Ternary rank 4096 | 181 MiB | 1,500 | 68.4% |
| Ternary rank 8192 | 274 MiB | 1,000 | 65.5% (rank 4096 at the same step: 65.0%) |

Multi-draft training (drafts 2 and 3 computed on the head's own outputs, 400 steps):

| Head | Draft 1 | Draft 2 | Draft 3 |
| :--- | ---: | ---: | ---: |
| Float | 70.3% → 70.2% | 24.9% → 52.4% | 12.8% → 43.2% |
| Ternary rank 4096 | 68.4% → 68.0% | 26.1% → 49.1% | 13.8% → 39.7% |

Engine, 12 prompts, tok/s with acceptance in brackets. Plain decoding: 35.3 tok/s.

| Head | 1 draft | 2 drafts | 3 drafts |
| :--- | ---: | ---: | ---: |
| Float, first-draft training | 44.0 (75.5%) | 36.9 (46.6%) | 31.9 (32.1%) |
| Float, multi-draft training | 43.7 (74.8%) | 42.3 (58.2%) | 37.6 (44.2%) |
| Ternary 4096, first-draft training | 44.2 (73.4%) | 38.5 (46.8%) | 32.4 (32.0%) |
| Ternary 4096, multi-draft training | 43.9 (73.4%) | 42.0 (56.6%) | 37.8 (42.8%) |
| Ternary 8192, first-draft training | 42.8 (71.2%) | 36.9 (44.5%) | 31.5 (30.9%) |

A further 1,200 steps of multi-draft training (1,600 in total):

| Head | Draft 1 / 2 / 3 agreement | 1 draft | 2 drafts | 3 drafts |
| :--- | ---: | ---: | ---: | ---: |
| Float | 71.3% / 56.6% / 48.5% | 43.0 (75.7%) | 41.8 (59.5%) | 38.9 (46.9%) |
| Ternary 4096 | 69.8% / 54.6% / 46.8% | 44.7 (75.6%) | 44.2 (59.6%) | 39.8 (45.3%) |

With the longer run the ternary head reaches the float head's acceptance, and 2 drafts is level with
1 draft (44.2 against 44.7 tok/s) without overtaking it.

### 5. Smaller findings

- **Ternary is not faster, it is smaller.** Per draft-and-verify cycle the ternary head is 1–4% faster
  than the float one. A plain step takes about 28 ms and a 1-draft cycle about 40 ms; the head's own
  format changes that by well under 1 ms.
- **On the original file, one draft is the fastest setting, because of verify cost.** Each extra token in
  the verify call costs about 6–9 ms there (22–33% of a single-token step); on the Q4_K_XL control it costs
  about 0.3–1.4 ms. Section 8 traces this to the weights still in the TQ1_0 layout and removes most of it.
  Measurements, cycle breakdown and kernel profile: `results/verify-cost.md`.
- **What a draft costs to produce.** Per cycle the engine calls the head once to catch up with the verify
  batch (1.5 ms) and once per draft (2.3–2.6 ms each, about 1 ms of it the 248k-row output projection);
  draft sampling is 0.03 ms. The ternary head costs 4.1 ms per 1-draft cycle against 5.7 ms for the float head.
- **A confidence threshold does not help in this engine.** `--spec-draft-p-min` raised acceptance
  (up to 95%) but lowered speed at every setting tried (30–38 tok/s against 42 without it).
- **History positions as training signal** (float head, 600 steps, scored on generated positions):
  history weight 0 gives 64.9%, 0.2 gives 65.6%, 1.0 gives 65.9%. Training on history only, with
  generated positions at weight zero, gives 59.0% (from 49.6% untrained). So text the model did not
  write is useful but clearly worse than its own output, in line with the EAGLE and SpecForge reports.
- **Context length does not matter to the head.** A head trained on short contexts scored the same on
  positions with about 5,200 tokens of history as on positions with about 1,000.
- **The trunk's hidden states are in a different basis from the stock model's.** A head fine-tuned
  against the stock embedding and output matrices scores 0.0% against the ternary file's own matrices
  until retrained.

### 6. Lookup drafting on top of the head (day 3)

The engine's n-gram drafters propose tokens by matching text already in the context, at no cost to
produce. Because a verify call of more than 8 tokens costs a flat ~258 ms on this engine (see
`results/verify-cost.md`), a long accepted lookup draft is very cheap per token. Ternary rank-4096
head, 12 general prompts and 6 copy-heavy edits ("add type hints and return the whole file",
"change two settings in this config", and similar), tok/s:

| Configuration | General prompts | Copy-heavy edits |
| :--- | ---: | ---: |
| Plain decoding | 35.7 | 35.4 |
| MTP head, 1 draft | 44.1 | 46.5 |
| `ngram-mod` alone | 35.4 | 78.3 |
| `ngram-simple`, 48-token drafts, alone | 31.3 | 75.4 |
| MTP head + `ngram-simple`, 48-token drafts | 36.2 | 79.5 |
| **MTP head + `ngram-mod`** (`--spec-type draft-mtp,ngram-mod`) | **43.0** | **81.6** |

- The two drafters cover different cases: the head gives the gain on ordinary generation, lookup gives
  the gain where the answer repeats the prompt. Together: 1.2x on general prompts and 2.3x on edits,
  with single edits between 67 and 134 tok/s.
- Fixed 48-token lookup drafts hurt general prompts; `ngram-mod` only drafts when it has a long match.
- `ngram-simple` with 3- or 7-token drafts never produced a draft in these runs, so short lookup drafts
  remain untested.
- A trunk file with an in-file head only loads when `--spec-type` includes `draft-mtp`; without it the
  loader rejects the extra tensors ("wrong number of tensors"). The baselines above use the plain trunk file.

### 7. Does more data still help?

The float head trained for the same 1,500 steps on fractions of the day-2 data, same held-out set:

| Training data | Generated tokens | Held-out agreement | Final training loss |
| :--- | ---: | ---: | ---: |
| 12.5% | 0.40M | 62.6% | 0.80 |
| 25% | 0.77M | 65.5% | 0.91 |
| 50% | 1.54M | 68.5% | 1.06 |
| 100% | 3.16M | 70.3% | 1.32 |
| 100%, 3,000 steps | 3.16M | 72.8% | 1.01 |

About 3 points per doubling, with the last step smaller until the full-data run is given twice the
steps (72.8%). The full-data run is also the one most short
of training steps (highest training loss), so the head is still data-limited at 3M tokens. At one
draft, though, a few more points of acceptance are worth only 1–3 tok/s on this engine.

### 8. Re-packing the old-layout weights: faster for every setting (day 3)

Profiling (`results/verify-cost.md`) showed that the cost of extra tokens in a call sits in one place:
the 47% of weights still stored as TQ1_0 (the main matrices whose width is not a multiple of 160, and
all ternary adapters) run through a generic kernel, while the TQ1_P table kernel batches almost for free.
`scripts/repack_tq1p.py` rewrites those tensors as TQ1_P: same trits, zero-padded to a multiple of 160,
with the per-block scale moved to a row-scale vector. It needs six lines of engine change
(`engine/ttf-mtp-head.patch`: two optional row-scale tensors in the loader, a pad in the graph) and no
kernel changes.

Trunk time per call (ms):

| Tokens per call | Original file | Re-packed |
| ---: | ---: | ---: |
| 1 | 27.1 | 25.5 |
| 2 | 33.2 | 28.3 |
| 3 | 40.4 | 30.4 |
| 4 | 47.3 | 31.5 |
| 8 | 82.5 | 55.1 |
| 16 | 258.6 | 265.9 |

Engine, tok/s (12 general prompts / 6 copy-heavy edits):

| Configuration | Original: general | Re-packed: general | Original: edits | Re-packed: edits |
| :--- | ---: | ---: | ---: | ---: |
| Plain decoding | 35.7 | 39.5 | 35.4 | 39.5 |
| MTP head, 1 draft | 44.1 | 50.9 | 46.5 | 52.8 |
| MTP head, 2 drafts | 43.3 | 54.0 | 46.4 | 58.0 |
| MTP head, 3 drafts | 39.8 | 54.9 | n/a | 57.1 |
| MTP head (2 drafts) + `ngram-mod` | n/a | 53.9 | n/a | 84.4 |
| `ngram-mod` alone | 35.4 | 38.6 | 78.3 | 77.4 |

- Plain decoding is 11% faster with no head at all, and multi-draft now pays: 2–3 drafts beat 1.
- Against the original plain speed, the re-packed model with the head is 1.5x on general prompts and
  2.4x on copy-heavy edits.
- Equivalence: on 1,103 positions through the small-batch path the re-packed file gives the same top
  token as the original on 99.82%; the original against itself at a different batch size gives 99.91%.
  On 15,409 positions through the prefill path it is 99.64%, with the differences at near-ties.
- Prefill is slightly slower (841 against 885 tok/s) and calls of more than 8 tokens are unchanged.
- The "original" columns for the head use 1,600-step multi-draft training and the 1-draft lookup
  combination from section 6; "n/a" means that combination was not run.

### 9. Where the limits are now

- **Three drafts is the sweet spot.** Four to eight drafts are slower (46.6 tok/s at 4, 9.1 at 8), because
  later drafts are rarely reached and the fast kernel takes at most 4 tokens per call.
- **Training the head deeper does not change that.** Positions 4 and 5 score about 40% untrained and 41–43%
  after 5-deep training.
- **Confidence-gated drafting is slower on this engine,** because a varying draft count forces a graph
  rebuild of about 12.6 ms on most cycles.

Details: `results/deep-drafting.md`.

### 10. Validation beyond the benchmark prompts

- **Speed holds** with thinking on, with sampling, and on real agent turns with about 5,000 tokens of
  history: the head gives 1.26x to 1.44x over re-packed plain decoding in every condition tested.
- **Accuracy is unchanged by the re-pack and the head** (60 general-knowledge and 60 IT-support questions):
  the re-pack changes no grades, the head changes 2 of 120.
- **The ternary model itself is less accurate on exact facts** than the 4-bit control of the same base
  model: 59/60 and 44/60 against 60/60 and 60/60. Ports, numbers and command syntax are where it slips.
- **Outputs are not word-for-word identical to plain decoding**, because the engine's arithmetic differs by
  call size and near-ties can fall either way.
- **With parallel slots the head is of limited use**: the slots share one verify call, and calls of more
  than 8 tokens are slow.

Details: `results/validation.md`.

### 11. Large batches: the engine fix for calls of more than 8 tokens (day 4)

Calls of more than 8 tokens paid a fixed ~262 ms, because the CUDA backend unpacked every ternary weight
to floats on each call. `engine/ttf-large-batch.patch` (about 145 added lines) lets the engine's batched
quantised kernel read TQ1_P weights directly.

| | Before | After |
| :--- | ---: | ---: |
| 16-token call | 266 ms | 57 ms |
| Prefill, 2,048 tokens | 831 tok/s | 1,192 tok/s |
| Head + `ngram-mod`, copy-heavy edits | 84.4 tok/s | 142.7 tok/s |
| Head + `ngram-mod`, general prompts | 53.9 tok/s | 54.2 tok/s |
| Head 3 drafts, 4 parallel slots | 29.5 tok/s total | 75.5 tok/s total |
| Plain, 8 parallel slots | 62.2 tok/s total | 69.4 tok/s total |

- **Head plus lookup is now the best single setting** (`--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2`).
- **This supersedes the "flat ~258 ms" figures** in sections 6, 9 and 10 and in `verify-cost.md`, which
  describe the engine before this patch.
- **Deep head drafting and the head at 4 or more slots are still not worth it**, for the reasons in
  sections 9 and 10; they are just no longer catastrophic.
- **Output:** the new path is as close to the old dense path as the engine's other paths are to each other
  (same top token on 28 of 28 prompts); accuracy on the 120 questions is unchanged.
- **The 4-bit control is still about 2x faster at prefill** (2,368 tok/s).

Details: `results/large-batch.md`.

### 12. A ternary DSpark drafter (day 5)

DSpark is a parallel drafter (five layers reading the target's hidden states, 7 tokens per pass) trained
for the full-precision model. Stock, on the ternary model, 0.2% of its drafts are accepted.

- **Cause:** the ternary model's embedding and output matrices are unrelated to the full-precision ones
  (cosine 0.000), and the drafter borrows the target's output head.
- **Fix:** give the drafter its own embeddings and output head, and fit one linear map from the ternary
  model's features to the ones the drafter expects. That alone gives control-level acceptance. Then
  ternarise the drafter and train it.

| Drafter | General | Edits | File |
| :--- | ---: | ---: | ---: |
| Plain decoding | 39.5 tok/s | 39.5 tok/s | |
| In-file MTP head + `ngram-mod` (section 11) | 54.2 | 142.7 | +0.5 GB |
| Stock DSpark, 7 drafts | 14.4 | 14.3 | 3.7 GB |
| Aligned float DSpark, 7 drafts | 58.1 | 123.5 | 5.2 GB |
| Fully ternary DSpark, 5 drafts | 60.6 | 112.4 | 1.0 GB |
| Fully ternary DSpark + `ngram-mod`, 5 drafts | 60.2 | 176.2 | 1.0 GB |

- **The fully ternary drafter is the fastest setting so far**: 1.53x plain on general prompts, 4.5x on edits.
- **Ternary costs about 15% of accepted drafts** against a float drafter (2.81 against 3.37 per block),
  which at 5 drafts does not show in speed.
- **A second engine patch** (`engine/ttf-table-kernel-8tok.patch`) lets the fast ternary kernel take 5 to 8
  tokens per call: 21% to 33% faster at those sizes, bit-identical output.
- **Short training** (8 hours in total), same 24 prompts. No weights are published.

Details: `results/dspark.md`.

Raw logs for all of the above are in `results/`:

| File | What it holds |
| :--- | :--- |
| `smoke-engine-test.txt`, `training-curves-day1.txt`, `head-to-head-engine.txt` | Sections 1 to 3 |
| `overnight-summary.txt`, `training-curves-overnight.txt`, `followup-summary.txt` | Sections 4 and 5 |
| `verify-cost.md`, `batch-size-bench-*.jsonl`, `profile/` | Verify cost: timings by batch size, cycle breakdown, timers, kernel profile |
| `lookup-drafting.txt` | Section 6 |
| `data-scaling.txt` | Section 7 |
| `repack-tq1p.txt` | Section 8 |
| `deep-drafting.md`, `deep-drafting-raw.txt` | Section 9 |
| `validation.md`, `validation-raw.txt` | Section 10 |
| `large-batch.md`, `large-batch-raw.txt` | Section 11 |
| `dspark.md`, `dspark-raw.txt` | Section 12 |

## How it works

- `engine/dump-nextn/`: `llama-dump-nextn`, a teacher-forced dumper. For every position it writes the
  token, the trunk's last hidden state (post output norm) and the trunk's top-32 next-token log-probabilities.
- `scripts/converse.py`: generates the training text. It renders prompts with `/apply-template` and uses
  the raw `/completion` endpoint so the exact token ids are logged. The target model writes every reply.
- `scripts/mtp_head.py`: a PyTorch replica of the engine's MTP graph, validated against the control model,
  and the multi-draft `unroll` (checked against a slow reference in `scripts/test_unroll.py`).
- `scripts/ttf_head.py`: the ternary head. Each matrix is `r * T(c x + m) + b`, plus a ternary low-rank
  adapter and a rank-16 float adapter, as in the trunk's TTF format 1. Trits come from latent float
  weights with a straight-through gradient; the adapter starts from an SVD of what ternarising lost.
- `scripts/train_mtp.py`: training. The loss is cross-entropy against the trunk's renormalised top-32.
- `scripts/build_infile.py`: writes the trunk GGUF again with the head appended as block 64
  (`block_count` 65, `nextn_predict_layers` 1). Run with `--spec-type draft-mtp` and no `-md`.
- `engine/ttf-mtp-head.patch`: one line so the loader applies TTF side tensors to the head's input
  projection, and the CMake entry for the dumper.
- `scripts/sessions_to_tasks.py`: turns Claude Code session logs into generation tasks and scrubs
  secrets first. It is pattern-based plus a list of literal strings; read its docstring before trusting it.
- `scripts/build_tasks.py`: rebuilds the public prompt set from its sources (not redistributed here).
- `scripts/overnight.sh`, `scripts/next.sh`: the unattended runs behind sections 4 and 5.
- `scripts/repack_tq1p.py`: re-packs a TTF GGUF's TQ1_0 tensors as TQ1_P (section 8). Needs the engine patch.
- `engine/ttf-large-batch.patch` (applied by `patch_bigbatch.py` then `patch_mmq_tq1p.py`): native batched-kernel
  tiles for TQ1_P, section 11
- `engine/ttf-mtp-head.patch`, `engine/patch_repack_engine.py`: the engine changes (six lines in the loader
  and graph, plus the build entry for the dumper). `engine/add_spec_timing.py` adds the optional timers.
- `scripts/bench12.py`, `scripts/bench_copy.py`: the 12 general prompts and the 6 copy-heavy edits.
- `scripts/lookup.sh`, `scripts/deep-draft.sh`, `scripts/deep5.sh`, `scripts/scale.sh`, `scripts/repack-test.sh`:
  the runs behind sections 6 to 9.

`docs/original-runbook.md` is the plan this started from, including the parts that turned out wrong.

## Caveats

- The ternary model is noticeably less accurate than a 4-bit quantisation of the same base model on exact
  facts (section 10). Nothing here changes that; this work only makes it faster.
- Everything was measured on one machine, one model and one engine build, greedy with thinking off.
- The engine comparison uses 12 prompts; differences of a point or two of acceptance are within noise.
- Most ablations are single runs.
- The scripts assume the directory layout of the original run (`~/ttf/...`) and are published as a
  record of what was done, not as a polished tool.

## Credits

- The ternary model, the TTF format and the engine patches: `oguzeldereli/ttf-home`.
- The stock MTP head and the control model: `unsloth/Qwen3.8-27B-GGUF`.
- The DSpark drafter (section 12): `RadixArk/Qwen3.8-27B-DSpark`; the DFlash and DSpark drafting support is the engine's.
- llama.cpp for the engine and its MTP implementation.
- Training ideas: EAGLE, EAGLE-3 (training-time test), Medusa, SpecForge and TorchSpec.
- Public prompt sources used to regenerate text (prompts only): `lightseekorg/kimi-mtp-dataset`,
  `Dahoas/rl-prompt-dataset`, `Usaid/Final_Prompt_Dataset`, `Prompt-Pool-Agent/datasets`,
  `chibbss/fitness-chat-prompt-completion-dataset`,
  `y1xing/natural_language_prompt_w_correct_ans_dataset_evaluation_instruct_dataset`.

## Licence

MIT for the code in this repository (see `LICENSE`). The model, the TTF format and the engine it patches
belong to their authors and carry their own licences.
