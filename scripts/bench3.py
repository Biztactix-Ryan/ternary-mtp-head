#!/usr/bin/env python3
"""Run the runbook's three test prompts against a llama-server and print tok/s and draft acceptance."""
import json, sys, urllib.request

PROMPTS = [
    "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings, then a short usage example.",
    "Explain in about 200 words how a refrigerator works.",
    "Write a bash script that finds the 10 largest files under a directory given as an argument, and prints sizes in human readable form.",
]

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "8097"
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    for i, p in enumerate(PROMPTS, 1):
        body = {
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 350, "temperature": 0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        t = r.get("timings", {})
        dn, da = t.get("draft_n"), t.get("draft_n_accepted")
        acc = f"{da}/{dn} ({100*da/dn:.1f}%)" if dn else "n/a"
        print(f"{label} prompt{i}: {t.get('predicted_per_second', 0):.1f} tok/s  n={t.get('predicted_n')}  drafts accepted {acc}", flush=True)

main()
