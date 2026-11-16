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

Result: draft acceptance at one draft went from 2–10% (stock head) to about 75%, and decoding
from 35.3 to about 44–45 tok/s (1.25x) on 2x RTX 5000 Ada, with a head that adds 181 MiB to the model
file instead of 810 MiB.

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
- **One draft is still the fastest setting, because of verify cost.** On the ternary model each extra token
  in the verify call costs about 6–9 ms (22–33% of a single-token step); on the Q4_K_XL control it costs
  about 0.3–1.4 ms. Producing each draft costs another 4–6 ms. With control-like batching the existing head
  would reach an estimated 51–58 tok/s instead of 44–45. Measurements and breakdown: `results/verify-cost.md`.
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

Raw logs for all of the above are in `results/`.

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

`docs/original-runbook.md` is the plan this started from, including the parts that turned out wrong.

## Caveats

- Everything was measured on one machine, one model and one engine build, greedy with thinking off.
- The engine comparison uses 12 prompts; differences of a point or two of acceptance are within noise.
- Most ablations are single runs.
- The scripts assume the directory layout of the original run (`~/ttf/...`) and are published as a
  record of what was done, not as a polished tool.

## Credits

- The ternary model, the TTF format and the engine patches: `oguzeldereli/ttf-home`.
- The stock MTP head and the control model: `unsloth/Qwen3.8-27B-GGUF`.
- llama.cpp for the engine and its MTP implementation.
- Training ideas: EAGLE, EAGLE-3 (training-time test), Medusa, SpecForge and TorchSpec.
- Public prompt sources used to regenerate text (prompts only): `lightseekorg/kimi-mtp-dataset`,
  `Dahoas/rl-prompt-dataset`, `Usaid/Final_Prompt_Dataset`, `Prompt-Pool-Agent/datasets`,
  `chibbss/fitness-chat-prompt-completion-dataset`,
  `y1xing/natural_language_prompt_w_correct_ans_dataset_evaluation_instruct_dataset`.

## Licence

MIT for the code in this repository (see `LICENSE`). The model, the TTF format and the engine it patches
belong to their authors and carry their own licences.
