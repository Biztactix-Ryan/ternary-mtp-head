# Wider validation (day 3)

Re-packed model (`scripts/repack_tq1p.py`) with the in-file ternary rank-4096 head, against plain decoding
and against the original file. Prompts are rendered with `/apply-template` and generated with `/completion`,
prompt cache off. Raw output: `validation-raw.txt`.

## Speed across conditions

"Agent turns" are 40 turns from 5 coding-agent sessions the head never trained on, with about 5,000 tokens
of history and tool definitions. tok/s:

| Condition | Original, plain | Re-packed, plain | Head, 3 drafts | Head (2 drafts) + `ngram-mod` |
| :--- | ---: | ---: | ---: | ---: |
| General (12), greedy | 35.5 | 38.8 | 57.2 | 55.0 |
| General, thinking on | 35.3 | 38.7 | 49.4 | 49.5 |
| General, temperature 0.7 | 35.3 | 38.8 | 52.3 | 52.9 |
| General, thinking on, temperature 0.7 | 35.2 | 38.7 | 48.7 | 46.9 |
| Agent turns, greedy | 34.5 | 37.7 | 54.3 | 52.3 |
| Agent turns, temperature 0.7 | 34.5 | 37.8 | 52.7 | 49.9 |
| Copy-heavy edits (6), greedy | 35.2 | 38.6 | 60.4 | 88.8 |
| Copy-heavy edits, temperature 0.7 | 35.3 | 38.6 | 59.8 | 87.5 |

- The head helps in every condition: from 1.26x (thinking on, sampled) to 1.44x on agent turns, against
  re-packed plain decoding.
- Thinking lowers acceptance (36% against 48% at 3 drafts); sampling costs 3–5 tok/s.
- Lookup only pays on near-verbatim edits; on real agent turns the head alone is faster.
- Prefill on the agent turns: 801 tok/s original, 746 re-packed, about 708 with the head loaded.
- 640 requests, none failed.

## Are the outputs the same as plain decoding?

Greedy runs, number of prompts with identical generated text:

| Comparison | General (12) | Thinking on (12) | Agent turns (40) | Edits (6) |
| :--- | ---: | ---: | ---: | ---: |
| Re-packed plain vs head, 3 drafts | 4 | 2 | 34 | 6 |
| Re-packed plain vs head + lookup | 5 | 2 | 31 | 6 |
| Original plain vs re-packed plain | 5 | 3 | 30 | 6 |

The outputs are not word-for-word identical. The engine takes different arithmetic paths for different call
sizes, so a near-tie between two tokens can fall the other way and the text diverges from there; the re-pack
alone shows the same level of divergence with no head involved, and the original model disagrees with itself
on 0.1–0.4% of tokens between call sizes. That each divergence starts at a near-tie was not checked directly;
the accuracy test below is the check on whether it matters.

## Answer accuracy

60 general-knowledge and 60 IT-support questions with checkable answers (`scripts/qa_accuracy.py`), greedy,
thinking off, graded by pattern match; every miss was read by hand and all are real errors.

| Configuration | General knowledge | IT support |
| :--- | ---: | ---: |
| Q4_K_XL control (same base model, 4-bit) | 60 / 60 | 60 / 60 |
| Ternary, original file | 59 / 60 | 44 / 60 |
| Ternary, re-packed | 59 / 60 | 44 / 60 |
| Ternary, re-packed + head, 3 drafts | 59 / 60 | 46 / 60 |
| Ternary, re-packed + head + lookup | 59 / 60 | 45 / 60 |

- **The re-pack changes no grades; the head changes 2 of 120**, both on questions where the plain model
  was wrong (near-ties falling the other way). 111–116 of 120 answers are identical text across configurations.
- **The ternary model itself is weaker on exact facts** than the 4-bit control: it keeps the shape of an
  answer and gets the detail wrong, for example LDAPS port "6361" (636), SMB port "4444" (445),
  `ipconfig flushdns` (`ipconfig /flushdns`), `gprefresh -force` (`gpupdate /force`), 11 FSMO roles (5).
  This is a property of the 1.6-bit model, not of anything in this repository, and it is the main thing to
  weigh against its memory saving (about 8 GB against about 19 GB for the control with its head).

## Parallel slots

Total tok/s with K concurrent requests (`scripts/bench_parallel.py`, 12 general prompts, 250 tokens each):

| Slots | Original, plain | Re-packed, plain | Re-packed + head, 1 draft | Re-packed + head, 3 drafts |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32.3 | 36.0 | n/a | n/a |
| 2 | 47.5 | 52.9 | 65.3 | 56.0 |
| 4 | 61.7 | 74.7 | 68.7 | 29.5 |
| 8 | 44.2 | 62.2 | n/a | n/a |

- The re-pack helps at every slot count, most at 8 (62.2 against 44.2).
- The slots' tokens share one call, so the call size is slots x (drafts + 1). Above 8 tokens the call takes
  the flat ~266 ms path: 4 slots at 3 drafts is 16 tokens and falls to 29.5 tok/s. With several slots, use
  fewer drafts (4 slots x 1 draft = 8 tokens) or no head; the head is mainly a single-stream gain until the
  large-batch path improves.
