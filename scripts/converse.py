#!/usr/bin/env python3
"""Conversation tool for collecting the ternary model's own generations (runbook section 8).

Renders prompts with /apply-template and generates with the raw /completion endpoint, so the
logged text and token ids are exactly what the model saw and produced.

  converse.py batch tasks.jsonl [more.jsonl ...]
  converse.py new [--system S] [--think] --user U
  converse.py say CONV_ID "next user message"
  converse.py stats

Environment: TTF_SERVERS (comma separated base URLs), TTF_SLOTS (slots per server),
TTF_DATA (output directory, default ~/ttf/data/gen).
"""
import argparse, collections, json, os, queue, sys, threading, time, uuid
import requests

SERVERS = os.environ.get("TTF_SERVERS", "http://127.0.0.1:8101,http://127.0.0.1:8102").split(",")
SLOTS = int(os.environ.get("TTF_SLOTS", "4"))
DATA = os.path.expanduser(os.environ.get("TTF_DATA", "~/ttf/data/gen"))
LOG = os.path.join(DATA, "conversations.jsonl")
CONVS = os.path.join(DATA, "convs")
N_PREDICT = 1024
MAX_PROMPT_TOKENS = 10800         # slot context is 12288; leave room for n_predict
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
_lock = threading.Lock()


def has_loop(tokens, window=20, times=3, stop_reason="limit"):
    """True for a runaway repetition: the completion ran to the token limit and one of the last few
    `window`-token windows occurs at least `times` times. (Any-window-repeats-anywhere also fires on
    ordinary code-heavy answers that quote the same lines three times, so it is not used.)"""
    if stop_reason != "limit" or len(tokens) < window * times:
        return False
    seen = collections.Counter(tuple(tokens[i:i + window]) for i in range(len(tokens) - window + 1))
    return any(seen[tuple(tokens[i:i + window])] >= times for i in range(len(tokens) - window - 5, len(tokens) - window + 1))


def strip_thinking(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def generate(server, conv):
    """One model turn for conv = {id, messages, tools, thinking, turn}. Returns the log record."""
    body = {"messages": conv["messages"], "chat_template_kwargs": {"enable_thinking": bool(conv.get("thinking"))}}
    if conv.get("tools"):
        body["tools"] = conv["tools"]
    r = requests.post(server + "/apply-template", json=body, timeout=120)
    r.raise_for_status()
    prompt = r.json()["prompt"]
    r = requests.post(server + "/completion", json={
        "prompt": prompt, "n_predict": N_PREDICT, "cache_prompt": True, "return_tokens": True, **SAMPLING,
    }, timeout=1800)
    r.raise_for_status()
    j = r.json()
    toks = j.get("tokens") or []
    rec = {
        "id": f"{conv['id']}-t{conv['turn']}", "conv": conv["id"], "turn": conv["turn"],
        "prompt": prompt, "completion": j["content"], "completion_tokens": toks,
        "stop_reason": j.get("stop_type"), "n_gen": j.get("tokens_predicted"),
        "n_prompt_tokens": j.get("tokens_evaluated"), "thinking": bool(conv.get("thinking")),
        "loop": has_loop(toks, stop_reason=j.get("stop_type")), "server": server,
    }
    with _lock:
        with open(LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def advance(server, conv, entry):
    """Append a user message or tool result, generate, append the assistant reply. Returns the record."""
    msg = {"role": "user", "content": entry} if isinstance(entry, str) else entry
    conv["messages"].append(msg)
    rec = generate(server, conv)
    conv["messages"].append({"role": "assistant", "content": strip_thinking(rec["completion"])})
    conv["turn"] += 1
    return rec


def new_conv(cid, system=None, tools=None, thinking=False):
    msgs = [{"role": "system", "content": system}] if system else []
    return {"id": cid, "messages": msgs, "tools": tools, "thinking": thinking, "turn": 0}


def save_conv(conv):
    with open(os.path.join(CONVS, conv["id"] + ".json"), "w") as f:
        json.dump(conv, f, ensure_ascii=False)


def cmd_batch(args):
    tasks = []
    for path in args.tasks:
        with open(path, encoding="utf-8") as f:
            tasks += [json.loads(l) for l in f if l.strip()]
    done = set()
    if os.path.exists(LOG):
        with open(LOG) as f:
            done = {json.loads(l)["conv"] for l in f if l.strip()}
    todo = [t for t in tasks if t["id"] not in done]
    print(f"{len(tasks)} tasks, {len(todo)} to run, servers {SERVERS} x {SLOTS} slots", flush=True)
    q = queue.Queue()
    for t in todo:
        q.put(t)
    stats = {"turns": 0, "gen": 0, "loops": 0, "limit": 0, "errors": 0}
    t0 = time.time()

    def worker(server):
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            conv = new_conv(t["id"], t.get("system"), t.get("tools"), t.get("thinking", False))
            conv["messages"] += t.get("messages", [])          # replayed history (session tasks)
            try:
                for entry in t["turns"]:
                    rec = advance(server, conv, entry)
                    with _lock:
                        stats["turns"] += 1
                        stats["gen"] += rec["n_gen"] or 0
                        stats["loops"] += rec["loop"]
                        stats["limit"] += rec["stop_reason"] == "limit"
                    # stop a conversation that looped, never finished thinking, or is near the context limit;
                    # a reply that merely hit the token cap stays in the history and the conversation goes on
                    unfinished_think = rec["thinking"] and "</think>" not in rec["completion"]
                    if rec["loop"] or unfinished_think or (rec["n_prompt_tokens"] or 0) + (rec["n_gen"] or 0) > MAX_PROMPT_TOKENS:
                        break
            except Exception as e:  # keep the batch going
                with _lock:
                    stats["errors"] += 1
                print(f"error in {t['id']}: {e}", flush=True)
            with _lock:
                dt = time.time() - t0
                print(f"[{dt:6.0f}s] {t['id']:<12} turns {stats['turns']} gen {stats['gen']} "
                      f"({stats['gen']/dt:.0f} tok/s) loops {stats['loops']} limit {stats['limit']} errors {stats['errors']}", flush=True)

    threads = [threading.Thread(target=worker, args=(s,)) for s in SERVERS for _ in range(SLOTS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("batch done", stats, flush=True)


def cmd_new(args):
    cid = "live-" + uuid.uuid4().hex[:8]
    conv = new_conv(cid, args.system, None, args.think)
    rec = advance(SERVERS[0], conv, args.user)
    save_conv(conv)
    print(f"CONV {cid}  [{rec['stop_reason']}, {rec['n_gen']} tok{', LOOP' if rec['loop'] else ''}]\n")
    print(rec["completion"])


def cmd_say(args):
    with open(os.path.join(CONVS, args.conv + ".json")) as f:
        conv = json.load(f)
    rec = advance(SERVERS[0], conv, args.message)
    save_conv(conv)
    print(f"CONV {conv['id']} turn {rec['turn']}  [{rec['stop_reason']}, {rec['n_gen']} tok{', LOOP' if rec['loop'] else ''}]\n")
    print(rec["completion"])


def cmd_stats(args):
    convs, turns, gen, ctx, loops, limit = set(), 0, 0, 0, 0, 0
    if os.path.exists(LOG):
        with open(LOG) as f:
            for l in f:
                if not l.strip():
                    continue
                r = json.loads(l)
                convs.add(r["conv"]); turns += 1
                gen += r["n_gen"] or 0; ctx += r["n_prompt_tokens"] or 0
                loops += bool(r["loop"]); limit += r["stop_reason"] == "limit"
    print(f"conversations {len(convs)}  turns {turns}  generated tokens {gen}  context tokens {ctx}  "
          f"total positions ~{gen + ctx}  loops {loops}  hit n_predict {limit}")


def main():
    os.makedirs(CONVS, exist_ok=True)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("batch"); p.add_argument("tasks", nargs="+"); p.set_defaults(fn=cmd_batch)
    p = sub.add_parser("new"); p.add_argument("--system"); p.add_argument("--think", action="store_true")
    p.add_argument("--user", required=True); p.set_defaults(fn=cmd_new)
    p = sub.add_parser("say"); p.add_argument("conv"); p.add_argument("message"); p.set_defaults(fn=cmd_say)
    p = sub.add_parser("stats"); p.set_defaults(fn=cmd_stats)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
