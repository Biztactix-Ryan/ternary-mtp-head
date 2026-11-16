# Runbook: train an MTP draft head for the ternary Qwen3.8-27B (smoke test)

Written 2026-11-16 on the 3090 box, to be executed on the machine with 2x RTX 5000 (32 GB each).
The reader is a Claude Code session on that machine with no access to the earlier conversation, so
everything it needs is here. Things marked **verified** were measured or read from source on the
3090 box. Things marked **design** are a plan that has not been run; expect to adjust them.

## 1. Goal and what "success" means

The ternary Qwen3.8-27B (TTF, 1.6 bits per weight) runs at about 31 tok/s on a 3090 with plain
decoding. Speculative decoding with an MTP head (`--spec-type draft-mtp`) should speed that up,
but the stock MTP head, trained against the full-precision model, is almost never right when it
reads the ternary model's hidden states.

The smoke test answers one question: **if the stock MTP head is fine-tuned on hidden states dumped
from the ternary model, does draft acceptance rise well above 3%?**

Reference numbers, all **verified** on the 3090 with the engine build in section 3, three prompts,
greedy, thinking off, 8k context:

| Target model | Drafting | tok/s | Drafts accepted |
|---|---|---|---|
| Ternary 27B with adapters | none | 30.7–30.9 | n/a |
| Ternary 27B with adapters | stock MTP, 3 drafts | 11.1–14.4 | 37/930, 8/654, 36/870 |
| Ternary 27B with adapters | stock MTP, 2 drafts | 16.8–18.2 | 39/618, 10/446, 39/590 |
| Ternary 27B with adapters | stock DSpark, 7 drafts | 6.9–7.0 | 4/2387, 1/1659, 6/2240 |
| Regular Q4_K_XL (control) | none | 42.2–42.3 | n/a |
| Regular Q4_K_XL (control) | stock MTP, 3 drafts | 74.0–94.4 | 250/295, 177/273, 99/108 |

Smoke test passes if the fine-tuned head gets first-draft agreement with the ternary trunk above
roughly 40% on held-out data (stock is under 10%), and the engine's own "draft acceptance" log line
moves accordingly. It does not need to beat 31 tok/s yet: on this engine an MTP draft-and-verify
step costs about 2.8x a plain step, so even at the Q4_K_XL acceptance rate the estimated ceiling is
about 1.27x. Speed is a later problem; acceptance is the question now.

## 2. Machine prerequisites

- NVIDIA driver plus the CUDA toolkit (`nvcc` on PATH; CUDA 12 or 13).
- `git`, `cmake`, a C++17 compiler, Python 3.11+, `uv` (or pip).
- About 60 GB free disk for the smoke test (models 30 GB, dumps 10–15 GB, checkpoints).
- Check first: `nvidia-smi`, `nvcc --version`, `df -h`, and confirm both cards are idle.

Use one card per process with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`.

## 3. Build the patched engine (**verified** procedure)

The ternary GGUF needs llama.cpp plus the author's 12 TTF patches. `ttf engine build --q38` does a
full-history clone of llama.cpp, which was extremely slow on the 3090 box's link. The shallow
variant below produces the same tree.

```bash
mkdir -p ~/ttf && cd ~/ttf
git clone --depth 1 https://github.com/oguzeldereli/ttf-home
cd ttf-home
uv venv .venv && uv pip install --python .venv/bin/python -e . huggingface_hub

# pinned upstream commit the q38 patches apply to (ENGINE_BASE_Q38 in ttfhome/cli.py)
D=~/ttf/ttf-home/engine-q38
SHA=ca89968a23a13e571a6b802a9891d22763153b61
git init -q "$D" && git -C "$D" remote add origin https://github.com/ggml-org/llama.cpp
git -C "$D" fetch -q --depth 1 origin $SHA
git -C "$D" checkout -q -B ttf FETCH_HEAD
git -C "$D" -c user.name=ttf -c user.email=ttf@localhost am -q ~/ttf/ttf-home/llama.cpp/q38/*.patch
git -C "$D" log --oneline | head -13      # expect 12 TTF commits on top of ca89968

export TTF_LLAMA_CPP_Q38=$D
.venv/bin/ttf engine build --q38          # sees 12/12 patches applied and just builds, CUDA
ls $D/build-cuda/bin                      # llama-server llama-bench llama-cli ... libllama.so
```

Notes:
- The ttf-home commit used on the 3090 box was `b20fa5c` (2026-11-15). If `ttfhome/cli.py` has a
  different `ENGINE_BASE_Q38` by the time this runs, use that SHA instead (fetch needs the full
  40-character hash; get it from `https://api.github.com/repos/ggml-org/llama.cpp/commits/<short>`).
- Always `export TTF_LLAMA_CPP_Q38=~/ttf/ttf-home/engine-q38` before using the `ttf` CLI. Calling
  the binaries in `build-cuda/bin` directly also works and is what the rest of this runbook does.

## 4. Downloads

```bash
cd ~/ttf/ttf-home
# ternary target, with adapters (8.2 GB). The head is specific to this file: the no-adapters file
# has different hidden states and would need its own head.
.venv/bin/hf download oguzeldereli/ttf-Qwen3.8-27B-GGUF \
  ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf SHA256SUMS --local-dir models/ttf-27b
# sha256 b9b6baa3789874b956309dc0df6f44de6e924c2f686473c3cff87839c79273a3

# stock MTP head, standalone file (1.37 GB). Starting point for fine-tuning and export template.
.venv/bin/hf download unsloth/Qwen3.8-27B-GGUF MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  --local-dir models/drafts/unsloth-27b
# sha256 50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e

# control target for validating the PyTorch head replica (17.7 GB). Recommended, see section 7.
.venv/bin/hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --local-dir models/q4
```

Python environment for training (separate from the engine's needs):

```bash
uv pip install --python .venv/bin/python torch numpy requests tqdm
# gguf-py comes from the engine tree, so reader/writer know the TTF tensor types:
uv pip install --python .venv/bin/python -e engine-q38/gguf-py
```

First check that the baseline reproduces before building anything:

```bash
B=engine-q38/build-cuda/bin
M=models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
MTP=models/drafts/unsloth-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
CUDA_VISIBLE_DEVICES=0 $B/llama-bench -m $M -p 512 -n 128 -ngl 99
CUDA_VISIBLE_DEVICES=0 $B/llama-server -m $M -ngl 99 -c 8192 -fa on --port 8097 \
  -md $MTP --spec-type draft-mtp
```

With the server up, send a chat completion and read `timings.draft_n` and
`timings.draft_n_accepted` in the JSON response, or the `draft acceptance = ...` line in the server
log. Stock head on the ternary model should show a few percent. Thinking is switched off per request
with `"chat_template_kwargs": {"enable_thinking": false}`.

## 5. How the MTP head works (**verified** by reading the engine source)

Sources: `engine-q38/src/models/qwen35.cpp` (`graph_mtp`, and the trunk graph's `h_nextn`),
`engine-q38/common/speculative.cpp` (`common_speculative_impl_draft_mtp`),
`engine-q38/src/llama-ext.h`.

**Trunk output used by the head.** `h[p]` is the trunk's last-layer hidden state at position p
*after* the final `output_norm` RMS norm (tensor named `h_nextn` in the trunk graph). Width 5120.
The engine exposes it with:

```c
llama_set_embeddings_nextn(ctx, true, /*masked=*/false);   // all positions in the batch
float * llama_get_embeddings_nextn_ith(ctx, i);            // row i of the last batch
```

**Head input and output.** At position p the head consumes the pair `(token x[p], h[p-1])`, that
is, the trunk hidden state shifted right by one, and predicts `x[p+1]`. This is the same token the
trunk would predict at p, but the head has not seen the trunk's `h[p]`. The head keeps its own KV
cache over all positions, so it is a causal one-layer transformer over the sequence of pairs.

**Head graph**, one block stored as `blk.64.*` in the MTP file:

```
e   = RMSNorm_enorm( token_embd[x[p]] )            # embedding comes from the MTP file itself
hn  = RMSNorm_hnorm( h[p-1] )
z   = eh_proj( concat(e, hn) )                     # 10240 -> 5120, embedding half first
a   = RMSNorm_attn_norm(z)
q_full = attn_q(a)                                 # 5120 -> 12288 = 24 heads x (256 q + 256 gate)
q, gate = per head: first 256 dims, last 256 dims
q   = RMSNorm_attn_q_norm(q)   (per head, width 256);  k = RMSNorm_attn_k_norm(attn_k(a))
v   = attn_v(a)                                    # k, v: 4 KV heads x 256 (grouped-query attention)
q, k = RoPE(q), RoPE(k)
o   = causal_attention(q, k, v, scale = 1/sqrt(256))      # 24 x 256 = 6144
o   = attn_output( o * sigmoid(gate) )             # 6144 -> 5120
z   = z + o
f   = RMSNorm_post_attention_norm(z)
z   = z + ffn_down( silu(ffn_gate(f)) * ffn_up(f) )       # 5120 -> 17408 -> 5120
g   = RMSNorm_shared_head_norm(z)                  # this is the head's own "h_nextn" output
logits = output.weight @ g                         # output matrix also from the MTP file
```

No biases anywhere. RMS epsilon 1e-6. RoPE: `rope.dimension_count = 64`, `freq_base = 1e7`,
`dimension_sections = [11, 11, 10, 0]`. I did not trace `build_rope` for this architecture, so read
it in the source before implementing; section 7's validation will catch a wrong RoPE.

**Multi-token drafting.** For Qwen there is a single head, reused. Draft step 1 feeds
`(x[p], h[p-1])` with the true trunk hidden state. Step 2 feeds `(drafted x[p+1], g[p])`, where
`g[p]` is the head's own output from step 1, at position p+1. Step 3 likewise. Earlier positions
in the head's KV cache were all computed from true trunk hidden states.

**Tensors in the stock MTP file** (18 in total, 2.97B parameters):

| Tensor | Shape (PyTorch order, out x in) | Stored as | Role in training |
|---|---|---|---|
| `token_embd.weight` | 248320 x 5120 | Q3_K | frozen, copy through unchanged |
| `output.weight` | 248320 x 5120 | Q3_K | frozen, copy through unchanged |
| `output_norm.weight` | 5120 | F32 | unused by the head graph; copy through |
| `blk.64.nextn.eh_proj.weight` | 5120 x 10240 | Q4_K | train |
| `blk.64.attn_q.weight` | 12288 x 5120 | Q6_K | train |
| `blk.64.attn_k.weight`, `attn_v.weight` | 1024 x 5120 | Q6_K | train |
| `blk.64.attn_output.weight` | 5120 x 6144 | Q6_K | train |
| `blk.64.ffn_up`, `ffn_gate` | 17408 x 5120 | Q4_K | train |
| `blk.64.ffn_down` | 5120 x 17408 | Q4_K | train |
| `blk.64.nextn.enorm`, `hnorm`, `shared_head_norm` | 5120 | F32 | train |
| `blk.64.attn_norm`, `post_attention_norm` | 5120 | F32 | train |
| `blk.64.attn_q_norm`, `attn_k_norm` | 256 | F32 | train |

Trainable total is about 425M parameters. GGUF lists dimensions reversed relative to PyTorch
(`5120, 12288` in the dump means a `Linear(5120, 12288)` weight of shape `[12288, 5120]`).

When the head is loaded from its own file with `-md`, the engine uses that file's `token_embd` and
`output.weight`, not the target's. So training must use the dequantised values of exactly those two
tensors, and the export must copy them byte for byte. `gguf.quants.dequantize` in the engine's
`gguf-py` handles Q3_K, Q4_K and Q6_K.

## 6. Component A: the hidden-state dumper (**design**)

A small C++ tool built inside the engine tree, `llama-dump-nextn`. Nothing like it exists yet.

Location: `engine-q38/examples/dump-nextn/` with a `CMakeLists.txt` copied from
`examples/embedding/CMakeLists.txt` (links `llama-common llama`), plus one `add_subdirectory` line in
`examples/CMakeLists.txt`. Model the batch handling on `examples/simple/simple.cpp` in this tree:
the pinned llama.cpp uses `llama_batch_ext_*` and `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE,
batch)`, not the older `llama_decode`. Include `../../src/llama-ext.h` for the nextn calls, as
`common/speculative.cpp` does.

Input: a JSONL file, one sequence per line:

```json
{"id": "conv12-turn3", "prompt": "<rendered context text>", "completion": "<raw generated text>"}
```

Behaviour per sequence:
1. Tokenise `prompt` (with special-token parsing on, BOS as the model wants) and `completion`
   (special-token parsing on, no BOS). Concatenate. Record `n_prompt`.
2. Truncate to the context size (8192 for the smoke test), keeping the tail.
3. Clear the memory for the sequence, then decode in batches of up to 512 tokens with the output
   flag set on every position and `llama_set_embeddings_nextn(ctx, true, false)`.
4. For every position write: the token id, `h[p]` (5120 floats, stored as float16), and the
   trunk's top-K next-token candidates (K = 32: token ids and log-probabilities, taken from
   `llama_get_logits_ith` after a log-softmax over the full vocabulary).

Output: shards of about 100k positions each, as raw little-endian arrays plus an index:

```
shard-0000.tok.i32     [N]          token ids
shard-0000.h.f16       [N, 5120]    trunk hidden states (post output_norm)
shard-0000.topk.i32    [N, 32]      trunk top-32 token ids
shard-0000.topk.f16    [N, 32]      matching log-probabilities
shard-0000.index.jsonl one line per sequence: {"id", "start", "length", "n_prompt"}
```

Storage is about 10.5 KB per position, so 1M positions is about 10.5 GB.

Checks before trusting the dumper:
- `argmax` of the written top-K at the last prompt position equals the first token the server
  generates greedily for the same prompt.
- Running the same file twice gives identical token and top-K files.
- Hidden-state rows are finite and their RMS is of order 1 (they are post-norm).

Run it once per card on half the input each. Expected speed is in the region of the 709 tok/s
prefill measured on the 3090; the 5000s may differ.

## 7. Component B: PyTorch replica of the head, and validating it (**design**)

`mtp_head.py`: a module implementing section 5's graph, with a loader that reads the stock MTP GGUF
through `gguf.GGUFReader`, dequantises every tensor to float32, and assigns them. Keep
`token_embd` and `output.weight` as frozen bf16 buffers on the GPU (about 2.5 GB each).

**Validate before any training.** A wrong RoPE or head split would silently give low agreement and
be mistaken for "ternary drift". The control is the regular model, where the stock head is known to
work:

1. Dump 50–100k positions from `models/q4/Qwen3.8-27B-UD-Q4_K_XL.gguf` with the dumper (any
   sensible chat-formatted text; the section 8 output is fine).
2. Run the replica with stock weights over those dumps: input `(x[p], h[p-1])`, compare
   `argmax(logits[p])` with the trunk's top-1 at p (`topk.i32[p, 0]`).
3. Expect high agreement, in the region of 80% on generated positions. The engine's measured
   3-draft acceptance on this model was 65–92%, and first-draft agreement should be at or above that.
   If the replica is far below, the replica is wrong: fix it before continuing.
4. Then run the same stock replica over a ternary dump. Expect low agreement (the engine measured
   about 3–4% acceptance averaged over 3 drafts; first-draft agreement may be somewhat higher).
   This number is the baseline the fine-tune has to beat.

## 8. Component C: the conversation tool, for the first ~1M tokens (**design**)

Purpose: have the ternary model generate text in realistic conversations, with Claude (the Claude
Code session on this machine) playing the user, and log exactly the token stream the model saw and
produced. The model's own output is the best-matched training text.

**Why raw completions.** The chat endpoint returns parsed messages, and the template may drop earlier
thinking blocks when re-rendering. The dumper needs the exact text. So the tool renders the prompt
itself and uses the raw endpoint:

1. `POST /apply-template` with the message list so far -> the rendered prompt string.
2. `POST /completion` with `{"prompt": ..., "n_predict": 1024, "temperature": 0.7, "top_p": 0.8,
   "top_k": 20, "cache_prompt": true}` -> raw generated text, including any `<think>` block.
3. Append one line to `data/gen/conversations.jsonl`:
   `{"id", "conv", "turn", "prompt", "completion", "stop_reason", "n_gen"}`. This is directly the
   dumper's input format.
4. To continue the conversation, strip the thinking block from the completion, add it as the
   assistant message, add the next user message, and go to 1.

Check on the first few calls that `/apply-template` output ends with the assistant generation
prefix, and that the completion stops at the end-of-turn token instead of running to `n_predict`.

**Servers.** One per card, several slots each:

```bash
CUDA_VISIBLE_DEVICES=0 $B/llama-server -m $M -ngl 99 -c 32768 -np 4 -fa on --port 8101 &
CUDA_VISIBLE_DEVICES=1 $B/llama-server -m $M -ngl 99 -c 32768 -np 4 -fa on --port 8102 &
```

`-c` is shared across slots (8192 each here). No draft model while generating data.

**`converse.py` commands:**

- `converse.py batch tasks.jsonl` — unattended. Each task is
  `{"id", "system", "turns": ["first user message", "follow-up", ...]}`. Runs all tasks across both
  servers and all slots, sending each scripted follow-up after the model replies.
- `converse.py new --system S --user U` -> prints a conversation id and the model's reply.
- `converse.py say CONV_ID "next user message"` -> prints the reply. This is the interactive mode
  where Claude reads the reply and writes a follow-up that reacts to it.
- `converse.py stats` -> conversations, turns, generated tokens, context tokens, loop count.

**Claude's role.**
1. Write `tasks.jsonl`: about 300 varied tasks with 2–4 scripted turns each. Aim the mix at how the
   model would be used: roughly half coding (write, debug, refactor, explain, shell and git, config
   files), the rest sysadmin and troubleshooting, planning, summarising pasted text, and general
   Q&A. Vary length, language, and whether thinking is on. Include some tasks whose system prompt
   defines tools, so tool-call syntax appears in the data, with the tool results scripted.
2. Run `batch` and let it finish unattended.
3. For 30–50 conversations, use `new`/`say` to do real back-and-forth, so some data has follow-ups
   that depend on what the model actually said. This is the expensive part in Claude usage; keep
   it small.

**Target for the smoke test:** about 1M total positions, of which at least 300–400k are generated
(completion) positions. Each turn is its own sequence, so later turns repeat earlier context; that
is acceptable here because the loss is weighted towards generated positions (section 9).

**Guard against loops.** The smaller Tiel-Coder model repeated itself until the token limit in 5 of
11 test turns; the 27B did not loop but once rambled to the limit. Cap `n_predict` at 1024, and have
the tool flag any completion where a 20-token window repeats three times; drop those from training.

Rough time: unknown on these cards. On the 3090 one stream ran at about 31 tok/s, so 400k generated
tokens is about 3.5 hours for one stream; two cards with 4 slots each should be several times
faster, but batched throughput on the ternary kernels has not been measured.

## 9. Component D: training (**design**)

`train_mtp.py`, single GPU, bf16 autocast, AdamW.

Data: memory-map the shards. Each sample is one sequence from the index. Hold out 5% of
*conversations* (not turns) for evaluation.

**Stage 1, first-draft only (do this first):**
- Input at p: `(tok[p], h[p-1])` for p >= 1. Standard causal attention over the sequence.
- Target at p: the trunk's top-32 distribution at p. Loss = cross-entropy against the trunk's
  renormalised top-32 probabilities, computed from the head's full-vocabulary logits. If memory is
  tight, compute the logits in chunks of positions.
- Position weights: 1.0 where `p >= n_prompt` (generated), 0.2 elsewhere.
- Starting hyperparameters (guesses): learning rate 1e-4 with 100 warmup steps then cosine decay,
  weight decay 0.01, gradient clip 1.0, about 16k positions per step (8 sequences of 2048), 150–300
  steps. 1M positions is about 60 steps per epoch.
- Metrics on held-out, generated positions only, every 25 steps: top-1 agreement with the trunk's
  top-1, and mean trunk-probability of the head's top-1. Log the stock head's values at step 0.

**Stage 2, unrolled drafting (only if stage 1 moves the number):**
- Pass A is stage 1's forward; keep its keys, values and outputs `g[p]`.
- Pass B: at each position q, input `(tok[q], g[q-1])`, attending to pass A's keys and values for
  positions < q and its own at q. Target: trunk top-32 at q. This is draft step 2.
- Pass C: input `(tok[q], g_B[q-1])`, attending to pass A for positions < q-1, pass B at q-1, own
  at q. This is draft step 3.
- Loss = step 1 + 0.7 x step 2 + 0.5 x step 3 (weights are a guess).

The second card can train a variant in parallel, for example a different learning rate, or freeze
everything except `eh_proj` and the two input norms to see how much a cheap input re-alignment
recovers on its own.

Expected memory: the trainable block with optimiser state is about 7 GB, frozen embedding and output
about 5 GB in bf16, plus activations. Should fit in 32 GB without chunking tricks; not measured.

## 10. Component E: export and the real test (**design**)

`export_mtp.py`: read the stock MTP GGUF with `GGUFReader`, write a new file with `GGUFWriter`:
- copy every metadata key as is (`engine-q38/gguf-py/gguf/scripts/gguf_new_metadata.py` shows how
  to copy keys and raw tensor data);
- copy `token_embd.weight`, `output.weight` and `output_norm.weight` raw, in their stored
  quantisation;
- write the trained `blk.64.*` matrices as F16 and the norm vectors as F32.

Check the round trip first: export the *untrained* replica and confirm the engine gives the same
acceptance as the stock file on the control model, to within noise. F16 instead of Q4_K/Q6_K should
make no visible difference.

Then the real test, on the ternary model, with prompts that were not in the training data:

```bash
$B/llama-server -m $M -ngl 99 -c 8192 -fa on --port 8097 \
  -md out/mtp-ternary-ft.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

Compare `draft acceptance` in the server log and tok/s against section 1's table, for
`--spec-draft-n-max` 1, 2 and 3. The three prompts used on the 3090 were:

1. "Write a Python class implementing an LRU cache with get and put, with type hints and
   docstrings, then a short usage example."
2. "Explain in about 200 words how a refrigerator works."
3. "Write a bash script that finds the 10 largest files under a directory given as an argument, and
   prints sizes in human readable form."

All with `max_tokens` 350, temperature 0, thinking off.

## 11. Order of work

1. Sections 2–4: build, download, reproduce the baseline and the stock-head acceptance numbers.
2. Dumper (section 6) and its checks.
3. Conversation tool (section 8); start `batch` generation running on both cards.
4. While it generates on one card: replica and control validation (section 7) on the other.
5. Dump the generated data from the ternary model.
6. Stage 1 training, watch held-out agreement.
7. Export, round-trip check, real engine test.
8. Report: the stock versus fine-tuned agreement and acceptance, tok/s at 1, 2 and 3 drafts, and
   anything in this runbook that turned out wrong.

Stop and report instead of pushing on if: the replica cannot be made to agree with the control
(step 4), or stage 1 leaves held-out agreement near the stock head's value after 300 steps.

## 12. After the smoke test (not part of this run)

- Bulk data: There is about a year of Claude Code sessions on several machines, under
  `~/.claude/projects/<project>/<session>.jsonl`. Entries with `type` `user` or `assistant` hold
  `message.content`, either a string or a list of `text`, `tool_use` and `tool_result` blocks. They
  can be converted to chat messages, rendered with `/apply-template`, and dumped teacher-forced at
  several hundred tokens per second. A cheaper way to get the model's own writing is to cut each
  session at an assistant turn and let the ternary model write that turn.
- Full run estimate: 5–20M positions, 500–2,000 steps. These are guesses to be revised from the
  smoke test's learning curve.
- The speed ceiling is the ternary engine's small-batch verification cost (about 2.8x a plain step
  for a 3-draft MTP step, against about 1.7x on Q4_K_XL). That is in the author's CUDA kernels and
  is worth raising with them alongside any acceptance result.
- The author's exporters drop the MTP block on purpose (`no_mtp = True` in
  `ttfhome/gguf_export.py` and `ttfhome/q38_export.py`). A positive result here is a good reason to
  ask them to keep and fine-tune it during recovery training.
