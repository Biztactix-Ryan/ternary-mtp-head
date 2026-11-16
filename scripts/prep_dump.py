#!/usr/bin/env python3
"""Filter the generation log for the dumper: drop looped completions and split into N parts by conversation.
usage: prep_dump.py conversations.jsonl OUT_PREFIX [parts=2]   -> OUT_PREFIX.0.jsonl, OUT_PREFIX.1.jsonl
"""
import json, sys
from converse import has_loop
src, prefix = sys.argv[1:3]
parts = int(sys.argv[3]) if len(sys.argv) > 3 else 2
outs = [open(f"{prefix}.{i}.jsonl", "w") for i in range(parts)]
convs, kept, dropped, pos, gen = {}, 0, 0, 0, 0
for l in open(src):
    if not l.strip():
        continue
    r = json.loads(l)
    # re-apply the current loop rule (early records were flagged with a looser one)
    if not r.get("completion_tokens") or has_loop(r["completion_tokens"], stop_reason=r.get("stop_reason")):
        dropped += 1
        continue
    if r["conv"].startswith("sess-"):          # cuts of one session share context: keep them in one group
        r["conv"] = r["conv"].rsplit("-", 1)[0]
    i = convs.setdefault(r["conv"], len(convs) % parts)
    outs[i].write(json.dumps({k: r[k] for k in ("id", "conv", "prompt", "completion", "completion_tokens")}, ensure_ascii=False) + "\n")
    kept += 1; pos += (r["n_prompt_tokens"] or 0) + (r["n_gen"] or 0); gen += r["n_gen"] or 0
print(f"kept {kept} sequences from {len(convs)} conversations, dropped {dropped}; about {pos} positions, {gen} generated")
