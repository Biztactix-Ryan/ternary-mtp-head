# Large batches: calls of more than 8 tokens

Measured on the re-packed model (`results/repack-tq1p.txt`), one RTX 5000 Ada. Raw numbers:
`large-batch-raw.txt`.

## The problem

Any call of more than 8 tokens cost a fixed ~262 ms before doing any work, then about 1.1 ms per token.
That covered prefill, long lookup drafts, deep head drafts, and every verify call once several parallel
slots were drafting.

A kernel profile of a 16-token call showed where it went:

| Share of GPU time | Kernel |
| ---: | :--- |
| 52% | `dequantize_block_tq1_p`: unpack every ternary weight to half-precision floats |
| 40% | cuBLAS half-precision matrix multiply over the unpacked weights |

The CUDA backend has three matrix paths. The fused table kernel takes 1 to 4 tokens and the generic
vector kernel 5 to 8. Above that, formats with a batched quantised kernel ("MMQ") use it; TQ1_P had none,
so it fell through to the dense path, which unpacks the whole weight on every call.

## The fix

Two steps, both in `engine/` as patch scripts, with the combined diff in `engine/ttf-large-batch.patch`.

1. **`patch_bigbatch.py`** (about 20 lines). The engine already expands TQ1_P to Q4_0 for batched
   `MUL_MAT_ID`. Plain `MUL_MAT` now takes the same route. This removed most of the cost but still
   expands the whole weight per call (58% of a 16-token call).
2. **`patch_mmq_tq1p.py`** (about 125 lines). TQ1_P becomes a native MMQ format. A TQ1_P block is five
   32-value digit planes, so although a 160-value block does not divide the kernel's 256-value tile, a
   plane does: the tile loader addresses the weight in planes, 8 per tile, and decodes each plane's trits
   to int8 directly into the tile. From there it is the existing int8 tile path with scale 1. Tile
   configurations are those of Q2_0. No extra memory, no per-call expansion.

`GGML_CUDA_TQ1P_NO_MMQ=1` selects step 1's route and adding `GGML_CUDA_TQ1P_DENSE=1` the original dense
path, for comparison.

## Result: time per call

| Batch | Original | Step 1 | Step 2 (native tiles) | Q4_K_XL control |
| ---: | ---: | ---: | ---: | ---: |
| 9 tokens | 263 ms | 120 ms | 56 ms | 42 ms |
| 16 | 266 ms | 112 ms | 57 ms | 43 ms |
| 32 | 266 ms | 116 ms | 58 ms | 45 ms |
| 64 | 276 ms | 135 ms | 74 ms | 50 ms |
| 128 | 311 ms | 176 ms | 108 ms | 69 ms |
| 512 | 602 ms | 471 ms | 414 ms | 217 ms |
| Prefill rate at 2,048 | 831 tok/s | 1,052 tok/s | 1,192 tok/s | 2,368 tok/s |

On 40 real agent turns averaging 5,080 prompt tokens, prefill went from 750 to 1,051 tok/s.

The remaining gap to the 4-bit control at large batches is mostly per-token: the TTF linear's scale,
shift, pad and adapter steps run as separate small operations above 4 tokens, and each weight has two
extra low-rank ternary products.

## Result: what it does for decoding

Single slot, tok/s (12 general prompts / 6 copy-heavy edits):

| Configuration | General, before | General, after | Edits, before | Edits, after |
| :--- | ---: | ---: | ---: | ---: |
| Plain | 39.5 | 39.5 | 39.5 | 39.0 |
| Head, 3 drafts | 54.9 | 54.1 | 57.1 | 58.4 |
| Head, 8 drafts | 9.1 | 33.0 | 9.0 | 33.6 |
| `ngram-mod` alone | 38.6 | 39.0 | 77.4 | 115.8 |
| **Head + `ngram-mod`, 2 head drafts** | 53.9 | **54.2** | 84.4 | **142.7** |
| Head + `ngram-mod`, 16 drafts | n/a | 27.5 | n/a | 94.7 |
| Head + `ngram-mod`, 16 drafts, p-min 0.6 | n/a | 32.1 | n/a | 111.9 |

- **Copy-heavy edits gained the most:** 142.7 tok/s with head plus lookup (84.4 before), 3.7x plain decoding. Long
  lookup drafts were the calls paying the fixed cost.
- **Head plus lookup is now the best single setting:** it matches the head alone on general prompts and
  is 2.4x faster on edits.
- **Deep head drafting is still not worth it:** 8 drafts went from 9.1 to 33.0 tok/s, still below plain.
  The later drafts are wrong too often.

Parallel slots, total tok/s:

| Slots | Plain, before | Plain, after | Head 1 draft, after | Head 3 drafts, before | Head 3 drafts, after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.0 | 36.4 | n/a | n/a | n/a |
| 2 | 52.9 | 56.6 | 68.1 | 56.0 | 55.5 |
| 4 | 74.7 | 80.8 | 73.7 | 29.5 | 75.5 |
| 8 | 62.2 | 69.4 | 71.8 | not run | 71.4 |

- The collapse at 4 slots with 3 drafts (29.5) is gone (75.5).
- The head still only pays off at 1 or 2 slots. At 4 slots plain decoding is ahead (80.8); at 8 they are
  level.

## Is the output the same?

No path on this engine is bit-identical to another, because the quantised kernels round activations to
8 bits and the summation order differs. The test is whether the new path sits further from the old dense
path than the engine's existing paths sit from each other. It does not:

| Pair | Same top token | Mean log-probability difference |
| :--- | ---: | ---: |
| Native tiles vs dense | 28/28 | 0.029 |
| Q4_0 expand (existing engine code) vs dense | 28/28 | 0.031 |
| Native tiles vs Q4_0 expand | 28/28 | 0.031 |
| Native tiles vs itself | 28/28 | 0 |

(Next-token distribution straight after prefill, 28 prompts of 26 to 1,397 tokens.)

- Greedy text against the dense path: identical on 4 of 12 general prompts and 28 of 40 agent turns; the
  rest diverge at a near-tie, as between the other paths (`validation.md`).
- Accuracy on the 120 questions is unchanged: 59/60 general knowledge, 46/60 IT support.

## Not tested

- Only the tensor-core tile layout used on this card was run. The patch also compiles a path for cards
  without it, which has not been executed.
- `MUL_MAT_ID` with TQ1_P now also takes native tiles; this model has none, so that is untested.
- One card, one model.
