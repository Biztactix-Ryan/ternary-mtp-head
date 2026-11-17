#!/usr/bin/env python3
"""Wider validation of a llama-server setup: three prompt suites, with thinking and sampling switchable.

  validate.py PORT LABEL SUITE [--think] [--temp T] [--n N] [--max-tokens M] [--out DIR]
  validate.py compare DIR LABEL_A LABEL_B        # how often two greedy runs produced the same text

Suites:
  general   the 12 prompts of bench12.py
  copy      the 6 copy-heavy edits of bench_copy.py
  sessions  held-out coding-agent turns (long history, tool definitions) from a task file made by
            sessions_to_tasks.py; only sessions the head never trained on are used

Prompts are rendered with /apply-template and generated with /completion, with the prompt cache off so
prefill is measured. One JSON line per request (text and timings) is written under --out.
"""
import argparse, hashlib, json, os, sys, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))


def prompts_from(script):
    ns = {}
    exec(open(os.path.join(HERE, script), encoding="utf-8").read().split("\nport = ")[0], ns)
    return ns["PROMPTS"]


def held_out(conv):
    return int(hashlib.md5(conv.encode()).hexdigest(), 16) % 20 == 0


def load_suite(a):
    if a.suite == "general":
        return [{"id": f"g{i:02d}", "messages": [{"role": "user", "content": p}]} for i, p in enumerate(prompts_from("bench12.py"), 1)]
    if a.suite == "copy":
        return [{"id": f"c{i:02d}", "messages": [{"role": "user", "content": p}]} for i, p in enumerate(prompts_from("bench_copy.py"), 1)]
    tasks = []
    for l in open(a.tasks, encoding="utf-8"):
        t = json.loads(l)
        if not held_out(t["id"].rsplit("-", 1)[0]):
            continue
        last = t["turns"][0]
        msgs = [{"role": "system", "content": t["system"]}] + t["messages"] + [{"role": "user", "content": last} if isinstance(last, str) else last]
        tasks.append({"id": t["id"], "messages": msgs, "tools": t.get("tools")})
    step = max(1, len(tasks) // a.n)
    return tasks[::step][:a.n]


def compare(d, la, lb):
    import glob
    for fa in sorted(glob.glob(os.path.join(d, la, "*.jsonl"))):
        fb = os.path.join(d, lb, os.path.basename(fa))
        if not os.path.exists(fb):
            continue
        A = {json.loads(l)["id"]: json.loads(l) for l in open(fa)}
        B = {json.loads(l)["id"]: json.loads(l) for l in open(fb)}
        ids = [i for i in A if i in B]
        same = sum(A[i]["content"] == B[i]["content"] for i in ids)
        frac = []
        for i in ids:
            x, y = A[i]["content"], B[i]["content"]
            k = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            frac.append(k / max(len(x), len(y), 1))
        print(f"{os.path.basename(fa)[:-6]:<24} {la} vs {lb}: identical text {same}/{len(ids)}, "
              f"average shared prefix {100 * sum(frac) / max(len(frac), 1):.0f}% of the text")


if len(sys.argv) > 1 and sys.argv[1] == "compare":
    compare(*sys.argv[2:5]); sys.exit(0)

ap = argparse.ArgumentParser()
ap.add_argument("port"); ap.add_argument("label"); ap.add_argument("suite", choices=["general", "copy", "sessions"])
ap.add_argument("--think", action="store_true"); ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--n", type=int, default=40); ap.add_argument("--max-tokens", type=int, default=350)
ap.add_argument("--tasks", default=os.path.expanduser("~/ttf/data/tasks2/sessions.jsonl"))
ap.add_argument("--out", default=os.path.expanduser("~/ttf/data/validate"))
a = ap.parse_args()
base = f"http://127.0.0.1:{a.port}"
variant = f"{a.suite}-{'think' if a.think else 'nothink'}-t{a.temp:g}"
os.makedirs(os.path.join(a.out, a.label), exist_ok=True)
out = open(os.path.join(a.out, a.label, variant + ".jsonl"), "w", encoding="utf-8")
sampling = {"temperature": a.temp, "seed": 1} if a.temp == 0 else {"temperature": a.temp, "top_p": 0.8, "top_k": 20, "seed": 1}
tn = tt = dn = da = pn = pt = 0
n_ok = n_err = 0
for t in load_suite(a):
    try:
        body = {"messages": t["messages"], "chat_template_kwargs": {"enable_thinking": a.think}}
        if t.get("tools"):
            body["tools"] = t["tools"]
        prompt = requests.post(base + "/apply-template", json=body, timeout=120).json()["prompt"]
        r = requests.post(base + "/completion", json={"prompt": prompt, "n_predict": a.max_tokens, "cache_prompt": False, **sampling}, timeout=1800).json()
        tm = r["timings"]
    except Exception as e:
        n_err += 1
        print(f"{a.label} {variant} {t['id']}: error {type(e).__name__}", flush=True)
        continue
    n_ok += 1
    tn += tm.get("predicted_n", 0); tt += tm.get("predicted_ms", 0.0)
    pn += tm.get("prompt_n", 0); pt += tm.get("prompt_ms", 0.0)
    dn += tm.get("draft_n") or 0; da += tm.get("draft_n_accepted") or 0
    out.write(json.dumps({"id": t["id"], "content": r["content"], "stop": r.get("stop_type"), "timings": tm}, ensure_ascii=False) + "\n"); out.flush()
acc = f"{da}/{dn} ({100 * da / dn:.1f}%)" if dn else "n/a"
print(f"{a.label:<16} {variant:<26} {1000 * tn / max(tt, 1):6.1f} tok/s  gen {tn:>6} tok  prefill {1000 * pn / max(pt, 1):6.0f} tok/s over {pn:>7} tok "
      f"(avg prompt {pn // max(n_ok, 1)})  drafts accepted {acc}  requests {n_ok} ok {n_err} failed", flush=True)
