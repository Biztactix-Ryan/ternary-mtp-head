#!/usr/bin/env python3
"""Twelve fixed prompts (the runbook's three plus nine more: code, shell, prose, reasoning, agent-style)
against a llama-server; prints per-prompt and overall tok/s and draft acceptance.
usage: bench12.py PORT LABEL"""
import json, sys, urllib.request

PROMPTS = [
    "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings, then a short usage example.",
    "Explain in about 200 words how a refrigerator works.",
    "Write a bash script that finds the 10 largest files under a directory given as an argument, and prints sizes in human readable form.",
    "Write a TypeScript function that groups an array of objects by a key and returns a Map, with a generic signature and two usage examples.",
    "Give me a systemd unit file for a Python web service that restarts on failure, runs as a dedicated user, and reads its environment from a file. Explain each line briefly.",
    "A ZFS pool shows one disk as DEGRADED with checksum errors. List the steps to diagnose it and decide whether to replace the disk.",
    "Write a SQL query that returns, for each customer, their most recent order and the total they have spent, given tables customers(id, name) and orders(id, customer_id, total, created_at).",
    "Summarise the trade-offs between RAID-Z2 and mirrored vdevs for an 8-disk pool in a short table followed by a recommendation.",
    "A train leaves at 9:40 and travels 150 km at 90 km/h, then waits 15 minutes, then travels 80 km at 60 km/h. When does it arrive? Show the working.",
    "Write a polite email to a client explaining that a server migration will be delayed by two days because of a faulty replacement disk, and offer a call.",
    "Refactor this into smaller functions and explain what you changed:\n\ndef process(items):\n    out = []\n    for i in items:\n        if i is None: continue\n        if isinstance(i, str):\n            i = i.strip()\n            if i == '': continue\n            try: i = float(i)\n            except ValueError: continue\n        if i < 0: i = 0\n        out.append(round(i, 2))\n    return sorted(set(out))",
    "Write a git workflow for keeping 12 local patches on top of an upstream project that is rebased onto a new upstream commit every few weeks.",
]

port = sys.argv[1] if len(sys.argv) > 1 else "8097"
label = sys.argv[2] if len(sys.argv) > 2 else ""
tn = tt = dn = da = 0
for i, p in enumerate(PROMPTS, 1):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 350, "temperature": 0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=600)).get("timings", {})
    n, ms = t.get("predicted_n", 0), t.get("predicted_ms", 0.0)
    tn += n; tt += ms; dn += t.get("draft_n") or 0; da += t.get("draft_n_accepted") or 0
    acc = f"{t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else "n/a"
    print(f"{label} p{i:02d}: {t.get('predicted_per_second', 0):5.1f} tok/s  accepted {acc}", flush=True)
print(f"{label} TOTAL: {1000 * tn / max(tt, 1):.1f} tok/s over {tn} tokens, drafts accepted "
      + (f"{da}/{dn} ({100 * da / dn:.1f}%)" if dn else "n/a"), flush=True)
