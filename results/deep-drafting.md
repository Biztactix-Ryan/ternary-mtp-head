# Drafting deeper than 3, and confidence-gated drafting

All on the re-packed model (`results/repack-tq1p.txt`), ternary rank-4096 in-file head, 12 general prompts.

## More drafts per cycle

| Drafts per cycle | tok/s | Drafts accepted | Accepted tokens in total |
| ---: | ---: | ---: | ---: |
| 3 | 54.9 | 47.7% | 2,171 |
| 4 | 46.6 | 38.0% | 2,222 |
| 5 | 41.6 | 31.7% | 2,258 |
| 6 | 36.9 | 26.9% | 2,272 |
| 8 | 9.1 | 20.9% | 2,300 |

- A draft only counts if every draft before it in the cycle was right, so the fourth to sixth drafts add
  about 100 accepted tokens out of 3,700 while each costs about 2.3 ms to produce.
- The TQ1_P table kernel takes at most 4 tokens per call (3 drafts plus the sampled token). Beyond that
  each token costs about 7 ms, and a call of more than 8 tokens takes the flat ~266 ms path.

## Does training the head deeper help?

The head was trained 3 drafts deep. A further 500 steps trained 5 deep (`--unroll 5`, loss weights
1 / 0.7 / 0.5 / 0.4 / 0.3). Held-out agreement at each draft position, given the earlier ones are correct:

| Draft position | Before | After 5-deep training |
| ---: | ---: | ---: |
| 1 | 69.8% | 69.2% |
| 2 | 54.6% | 54.1% |
| 3 | 46.8% | 47.0% |
| 4 | 41.3% | 43.2% |
| 5 | 38.1% | 41.0% |

Positions 4 and 5 were already at about 40% without being trained, and training moved them by 2–3 points.
In the engine nothing changed: 56.4 / 46.0 / 42.2 tok/s at 3 / 4 / 5 drafts, with the same accepted counts
to within a few tokens. Each step reads the head's own previous output instead of the trunk's hidden
state, and that loss compounds; depth of training is not the limit.

## Confidence-gated drafting (`--spec-draft-p-min`)

| Setting | tok/s | Drafts accepted |
| :--- | ---: | ---: |
| 3 drafts, no threshold | 54.9 | 47.7% |
| 3 drafts, threshold 0.5 | 37.3 | 70.9% |
| 6 drafts, threshold 0.5 | 35.6 | 61.3% |
| 6 drafts, threshold 0.7 | 33.8 | 80.0% |
| 8 drafts, threshold 0.8 | 34.8 | 86.7% |

The threshold raises acceptance but lowers speed. The check itself is free (draft sampling measures
0.03 ms with or without it). The cost is on the trunk: a varying number of drafts makes the verify call's
size change between cycles, and the engine rebuilds its compute graph whenever the size changes.
Forcing a rebuild on every call (`LLAMA_GRAPH_REUSE_DISABLE=1`) shows what that costs: plain decoding
falls from 39.5 to 26.4 tok/s, about 12.6 ms per trunk call, and 3 drafts from 54.9 to 41.7.
Disabling CUDA graphs did not help the threshold runs, so re-capture is not the cause.
