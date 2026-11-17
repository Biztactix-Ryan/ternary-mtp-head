# A ternary DSpark drafter for the ternary model

Measured on the re-packed ternary model (`results/repack-tq1p.txt`) with the large-batch patch, one
RTX 5000 Ada per run. Raw numbers: `dspark-raw.txt`. No weights are published here.

DSpark (`RadixArk/Qwen3.8-27B-DSpark`) is a parallel drafter: five transformer layers that read the
target's hidden states at five depths and predict a block of 7 tokens in one pass. It was trained for the
full-precision model.

## Result

12 general prompts / 12 copy-heavy edits, tok/s. Plain decoding on this model is 39.5.

| Drafter | Drafts | General | Edits | File |
| :--- | ---: | ---: | ---: | ---: |
| In-file MTP head + `ngram-mod` (section 11) | 2 | 54.2 | 142.7 | +0.5 GB |
| Stock DSpark | 7 | 14.4 | 14.3 | 3.7 GB |
| Aligned (one regression, no training) | 7 | 58.1 | 123.5 | 5.2 GB |
| Float, fine-tuned 1.3 h | 7 | 60.5 | 125.4 | 5.2 GB |
| Ternary layers, 4-bit embeddings and output head | 5 | 60.4 | 113.2 | 2.2 GB |
| **Fully ternary** | 5 | **60.6** | 112.4 | **1.0 GB** |
| **Fully ternary + `ngram-mod`** | 5 | 60.2 | **176.2** | 1.0 GB |

For comparison, the 4-bit control with the stock drafter: 62.8 / 120.6 (plain 28.7).

- The fully ternary drafter is 1.53x plain decoding on general prompts and 4.5x on edits.
- It matches the float drafter on general prompts at a fifth of the size.
- All rows from "Aligned" down use the 8-token table kernel below.

## Why the stock drafter scores 0.2%

The ternary model's `token_embd.weight` and `output.weight` are not a quantisation of the full-precision
ones: their cosine similarity with the originals is 0.000. The model lives in a different internal basis.
The drafter has no output head of its own; it borrows the target's. With the ternary target's head it
predicts noise. Training from that start was slow: 1.0 accepted drafts per block after 3,500 steps.

## The fix

1. **The drafter carries its own embeddings and output head**, taken from the full-precision family (here
   the 4-bit control file). The engine's DSpark loader already accepts `token_embd.weight` and
   `output.weight` in the draft file.
2. **One linear map aligns the features** (`scripts/dspark_align.py`). The drafter's first layer is a
   25,600 -> 5,120 projection of the five trunk layers. Ridge regression over 507,100 positions fits a new
   projection that maps the ternary model's features to what the stock projection produces from the
   control's features on the same text (held-out R² 0.63, cosine 0.88). No gradient training.
3. **Ternarise and train** (`scripts/dspark.py train --tern`). All matrices become TQ1_0 (256 trits and
   one scale per block, 1.69 bits per weight), trained with a straight-through estimator against the
   trunk's top-32 distribution. Finally the embeddings and output head are ternarised and frozen
   (`--tern-io both`) and the layers trained against them for another hour.

## Held-out agreement

Accepted drafts per block, on 60 held-out sequences (5,489 blocks). Row 1 is the first drafted token.

| Drafter | Row 1 agreement | Per 7-block | Per 15-block |
| :--- | ---: | ---: | ---: |
| Stock, ternary trunk and its own head | 0.8% | 0.01 | 0.01 |
| Stock on the 4-bit control (reference) | 84% | 3.29 | 4.27 |
| Aligned | 80.5% | 3.12 | 4.02 |
| Float fine-tune (F, 1.3 h) | 82.5% | 3.37 | 4.45 |
| Ternary layers (G, 4.4 h) | 77.7% | 2.88 | 3.45 |
| Ternary layers, 6 deep (H, 2.8 h from F) | 78.0% | 2.88 | 3.46 |
| G with output head ternarised, untrained | 72.5% | 2.54 | 3.01 |
| G with head and embeddings ternarised, untrained | 71.6% | 2.49 | 2.95 |
| Ternary output head, layers retrained 1 h (J) | 77.6% | 2.82 | 3.37 |
| Fully ternary, layers retrained 1 h (I) | 77.2% | 2.81 | 3.37 |

- Ternary layers cost about 15% of accepted drafts against float.
- Ternarising the output head costs 0.34 untrained and 0.06 after an hour of retraining the layers.
- A sixth layer reached the 5-layer result in less training time but did not pass it. In the engine the
  two are within noise.
- G and H were still improving slowly when stopped.

## Draft length

| Drafter | 5 drafts | 7 drafts | 15 drafts (block 16) |
| :--- | ---: | ---: | ---: |
| Float F, general | | 60.5 | 49.0 |
| Float F, edits | | 125.4 | 151.3 |
| Ternary G, general | 60.4 | 54.2 | 36.4 |
| Ternary G, edits | 113.2 | 115.6 | 96.1 |
| Fully ternary I, general | 60.6 | 52.9 | |
| Fully ternary I, edits | 112.4 | 112.7 | |

The ternary drafter's later rows are weaker, so 5 drafts is its best general setting. Long blocks only pay
for the float drafter, and only on edits; `ngram-mod` covers edits better (176 to 183 tok/s).

## The 8-token table kernel

A 7-draft cycle verifies 8 tokens, which took the generic vector kernel (calls of 5 to 8 tokens).
`engine/patch_tq1p_cols8.py` (diff: `engine/ttf-table-kernel-8tok.patch`) lets the fused table kernel take
up to 8: the weight stream is read and decoded once per row, and the activations are processed in chunks
of four columns. `GGML_CUDA_TQ1P_COLS4=1` restores the old limit.

| Tokens per call | Before | After |
| ---: | ---: | ---: |
| 4 | 119.0 tok/s | 118.0 tok/s |
| 5 | 119.6 | 144.7 |
| 6 | 130.8 | 166.0 |
| 7 | 134.1 | 175.4 |
| 8 | 143.6 | 191.6 |
| 9 | 158.2 | 158.2 |

- Output is bit-identical to the old path on 26 prompts of 5 to 8 tokens.
- The aligned drafter at 7 drafts went from 50.1 to 58.1 tok/s (edits 103.0 to 123.5).

## Reproducing

| Step | Script |
| :--- | :--- |
| Stock baselines | `scripts/dspark-base.sh` |
| Feature dumps (inputs of trunk layers 6, 20, 34, 48, 62) | `scripts/dspark-dump.sh`, `llama-dump-nextn --layers` |
| Alignment | `scripts/dspark_align.py` |
| Training, evaluation, GGUF export | `scripts/dspark.py` |
| Schedules used | `dspark-train3.sh`, `dspark-window0.sh`, `dspark-final.sh`, `dspark-io-train.sh`, `dspark-io-final.sh` |
| Engine test | `scripts/dspark-test.sh` |

Training data: 2,400 public-prompt sequences generated by the ternary model itself (122 GB of features),
5% held out. Control features for the alignment: the first 600 of the same sequences through the 4-bit
control. Training needs a whole 32 GB card.

## Caveats

- **Short training.** Everything fits in one 8-hour window; the longest run is 4.4 hours.
- **Same 24 prompts** as the earlier sections; acceptance on other workloads was not measured.
- **Not tried:** other trunk layer sets (needs a new dump), a wider drafter, training the output head
  itself rather than only the layers under it, parallel slots.
- **The drafter depends on a full-precision-family output head** for its starting point. The alignment
  needs features from a model in the original basis.
- **Licence:** the DSpark checkpoint's licence is listed as "other"; check it before distributing
  anything derived from it.
