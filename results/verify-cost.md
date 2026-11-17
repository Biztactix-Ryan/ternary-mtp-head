# Where the draft-and-verify cycle time goes

Speculative decoding verifies the drafts by running the trunk on a small batch: the sampled token plus
1, 2 or 3 drafted tokens. This measures how the trunk's time per call grows with that batch size.

## Trunk time per call by batch size

`llama-bench -n 0 -p <batch> -r 10 -fa 1 -ngl 99`, one RTX 5000 Ada per model, engine build `720afe4`
(llama.cpp `ca89968` + the 12 TTF patches). Median of 10 repetitions; each repetition processes
`<batch>` tokens in one call from an empty context.

| Tokens per call | Ternary TQ1_P: ms per call | Q4_K_XL control: ms per call |
| ---: | ---: | ---: |
| 1 | 27.1 | 34.4 |
| 2 | 33.2 | 35.8 |
| 3 | 40.4 | 36.1 |
| 4 | 47.3 | 36.5 |
| 6 | 65.3 | 37.4 |
| 8 | 82.5 | 38.6 |
| 16 | 258.6 | 41.4 |
| 32 | 257.1 | 43.5 |
| 64 | 267.3 | 48.1 |

- **On the ternary model every extra token in a small batch costs about 6–9 ms**, which is 22–33% of a
  single-token step. On the control the same extra token costs about 0.3–1.4 ms.
- At one token the ternary model is the faster of the two (27.1 against 34.4 ms). By four tokens it is
  30% slower (47.3 against 36.5 ms).
- **There is a cliff between 8 and 16 tokens per call** on the ternary model: 82.5 ms at 8, about 258 ms
  at 16, then flat to 64. That looks like a switch from the per-token path to the prefill path, which
  has a large fixed cost. Batch sizes 9–15 were not measured.

Raw records: `batch-size-bench-ttf.jsonl`, `batch-size-bench-q4.jsonl`.

## Breakdown of a cycle with the ternary in-file head

Cycle times come from the 12-prompt engine runs of the rank-4096 ternary head after 1,600 steps of
multi-draft training (`followup-summary.txt`): total time divided by the number of draft-and-verify cycles.
The last column is what is left after subtracting the trunk call above. It was not profiled; it
contains the head's forward passes, the output projection and sampling for each draft, and per-call overhead.

| Drafts | Measured cycle | Trunk verify call | Remainder | Remainder per draft |
| ---: | ---: | ---: | ---: | ---: |
| 0 (plain) | 28.3 ms | 27.1 ms (1 token) | 1.2 ms | n/a |
| 1 | 39.4 ms | 33.2 ms (2 tokens) | 6.2 ms | 6.2 ms |
| 2 | 49.8 ms | 40.4 ms (3 tokens) | 9.4 ms | 4.7 ms |
| 3 | 59.6 ms | 47.3 ms (4 tokens) | 12.3 ms | 4.1 ms |

So each extra draft costs roughly 6–7 ms of trunk verification plus 4–6 ms for producing the draft.
Both parts matter; the verification part is the one that is specific to the ternary kernels.

## What cheaper verification would be worth

An estimate, not a measurement: keep the measured remainder and acceptance, but let extra tokens in
the verify call cost what they cost on the Q4_K_XL control (+1.4, +1.7 and +2.1 ms over one token).

| Drafts | Measured tok/s | Estimated tok/s with control-like batching |
| ---: | ---: | ---: |
| 1 | 44.7 | about 51 |
| 2 | 44.2 | about 58 |
| 3 | 39.8 | about 57 |

Plain decoding is 35.3 tok/s, so that would move the speedup from 1.25x to roughly 1.6x with the head
that already exists, and would make 2 drafts clearly better than 1.

## Why, from reading the kernels (not profiled)

These are readings of `ggml/src/ggml-cuda` in the patched engine, matched against the timings above.
They have not been confirmed with a profiler.

- **1–4 tokens: a dedicated matrix-vector kernel.** `mmvq-tq1p.cuh` is a "TQ1_P matrix-vector product for a
  few tokens (ncols_dst <= 4), table decoded", fused with the TTF scale, bias and adapter steps. Its header
  notes that decoding trits is compute-bound, not memory-bound. That fits the timings: a Q4_K step is
  dominated by reading 17 GB of weights once, so extra tokens ride along almost free, while the ternary
  step reads less than half as much and spends its time on decode and dot-product arithmetic, part of which
  repeats per token (about 6–7 ms each).
- **5–8 tokens: the generic path.** The fused kernel only accepts up to 4 tokens. Beyond that the products
  go through the generic `mul_mat_vec_q` kernels (limit `MMVQ_MAX_BATCH_SIZE` = 8) without the fusion,
  which matches the higher 8–9 ms per extra token.
- **More than 8 tokens: no matrix-matrix kernel for dense TQ1_P.** The source says "TQ1_P has no MMQ tile
  (its 160-value blocks do not divide MMQ_ITER_K)". Batched expert products are expanded to Q4_0 and use
  that MMQ kernel, but this 27B model is dense, and a flat cost of about 258 ms from 16 to 64 tokens looks
  like every weight being dequantised and multiplied with cuBLAS on each call.

The same curve explains two other observations from this work: 8 parallel generation slots gave only
about 3x the single-stream throughput, and a 4-draft setting would fall off the fused path.

## Measured: what producing a draft costs

The "remainder" column above was a subtraction. This is the direct measurement, with timers added to
the engine's MTP driver (`engine/add_spec_timing.py`, enabled with `TTF_SPEC_TIMING=1`; each timer
includes a device synchronise). 12 general prompts, in-file heads after 1,600 steps of multi-draft training.

Per cycle the driver makes one **catch-up** call on the head (it replays the tokens of the trunk's
verify batch through the head, without logits, to keep the head's cache in step) and then one
**draft** call per drafted token (one token, with logits).

| Head | Drafts | Catch-up call | Each draft call | Draft sampling | Head time per cycle |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Ternary rank 4096 | 1 | 1.50 ms (2.3 tokens) | 2.60 ms | 0.03 ms | 4.1 ms |
| Ternary rank 4096 | 2 | 1.83 ms (3.3 tokens) | 2.43 ms | 0.03 ms | 6.7 ms |
| Ternary rank 4096 | 3 | 2.03 ms (4.3 tokens) | 2.33 ms | 0.03 ms | 9.0 ms |
| Float F16 | 1 | 2.20 ms (2.3 tokens) | 3.47 ms | 0.03 ms | 5.7 ms |

- A 1-draft cycle of 39.4 ms is therefore about 33.2 ms of trunk verification, 4.1 ms of head work
  (1.5 catch-up + 2.6 draft) and about 2 ms of server-side work, of which about 1.2 ms is also present
  in plain decoding (28.3 ms cycle against a 27.1 ms trunk call).
- The ternary head costs 1.6 ms less per cycle than the float head (4.1 against 5.7 ms).
- A draft call costs about 1.1 ms more than a catch-up call that carries more tokens, so the output
  projection over the 248k vocabulary is roughly a millisecond of each draft.
- From reading the driver: the first token of each catch-up call is the same (token, hidden state)
  pair the previous draft call already ran, so part of the catch-up is repeated work, and folding the
  catch-up into the next draft call would save most of a call per cycle. That is untested.
