#!/usr/bin/env python3
"""Turn Claude Code session logs into generation tasks for converse.py, scrubbing secrets first.

Runs on the machine that holds the sessions; only its scrubbed output is meant to be copied anywhere.

  sessions_to_tasks.py PROJECTS_DIR OUT_DIR [--cuts 40] [--skip-project NAME ...] [--extra extra.txt]

Each task is one point in a session where the assistant was about to reply: the scrubbed history up to
there (tail, about 5k tokens) plus generic tool definitions. The local model writes that one turn.
Thinking blocks and images are dropped. Output: OUT_DIR/tasks-sessions.jsonl and OUT_DIR/scrub-report.txt
(counts and masked samples only, never a secret's value).

--extra: a text file of literal strings to always redact, one per line (your own passwords, names, hosts).
"""
import argparse, collections, glob, hashlib, hmac, json, math, os, random, re, secrets

ap = argparse.ArgumentParser()
ap.add_argument("projects"); ap.add_argument("out")
ap.add_argument("--cuts", type=int, default=40, help="max generation points per session")
ap.add_argument("--skip-project", action="append", default=[])
ap.add_argument("--extra")
ap.add_argument("--ctx-chars", type=int, default=20000); ap.add_argument("--tool-chars", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)
KEY = secrets.token_bytes(32)          # per-run key: replacements are consistent within a run, not reversible
counts = collections.Counter(); samples = collections.defaultdict(list)

HEX = "0123456789abcdef"
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def fake(s, alphabet):
    """same-length stand-in drawn from `alphabet`, keeping - _ . / + = in place so the text still looks natural"""
    d = hmac.new(KEY, s.encode(), hashlib.sha256).digest()
    out, i = [], 0
    for ch in s:
        if ch.isalnum():
            out.append(alphabet[d[i % 32] % len(alphabet)]); i += 1
        else:
            out.append(ch)
    return "".join(out)


def note(kind, m, repl):
    counts[kind] += 1
    if len(samples[kind]) < 4:
        s = m.string
        samples[kind].append((s[max(0, m.start() - 30):m.start()] + "[" + kind + f" x{m.end() - m.start()}]" + s[m.end():m.end() + 15]).replace("\n", " "))
    return repl


def entropy(s):
    c = collections.Counter(s)
    return -sum(v / len(s) * math.log2(v / len(s)) for v in c.values())


def sub(kind, pattern, text, repl, flags=0):
    return re.sub(pattern, lambda m: note(kind, m, repl(m) if callable(repl) else repl), text, flags=flags)


EXTRA = []
if a.extra and os.path.exists(a.extra):
    EXTRA = [l.strip("\r\n") for l in open(a.extra, encoding="utf-8") if l.strip()]

PRIVATE_NET = re.compile(r"^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|169\.254\.|0\.|255\.|22[4-9]\.|23\d\.)")


def scrub(t):
    if not t:
        return t
    for lit in EXTRA:                                  # case-insensitive literal match
        t, n = re.subn(re.escape(lit), "REDACTED", t, flags=re.I)
        counts["extra literal"] += n
    t = sub("private key block", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)", t,
            "-----BEGIN PRIVATE KEY-----\nREDACTED\n-----END PRIVATE KEY-----", re.S)
    # vendor token formats
    t = sub("vendor token", r"\b(sk-ant-[\w-]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|"
            r"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|xox[baprs]-[\w-]{10,}|AIza[\w-]{30,}|tskey-[\w-]{10,}|glpat-[\w-]{15,}|"
            r"npm_[A-Za-z0-9]{30,}|pypi-[\w-]{30,}|SG\.[\w-]{16,}\.[\w-]{16,})", t, lambda m: fake(m.group(0), B64))
    t = sub("jwt", r"\beyJ[\w-]{8,}\.eyJ[\w-]{8,}\.[\w-]{8,}", t, lambda m: fake(m.group(0), B64))
    t = sub("password hash", r"\$(?:1|2[aby]|5|6|y|argon2\w*)\$[./A-Za-z0-9$=,+-]{20,}", t, lambda m: fake(m.group(0), B64))
    t = sub("url credentials", r"(?<=://)([^/\s:@]+):([^/\s@]+)@", t, "user:REDACTED@")
    t = sub("auth header", r"(?i)\b(authorization\s*[:=]\s*(?:bearer|basic|token)\s+|bearer\s+)([A-Za-z0-9._~+/=-]{12,})", t,
            lambda m: m.group(1) + "REDACTED")
    # name = value / name: value / --name value, where the name says it is a secret
    name = r"(?:pass(?:word|wd|phrase)?|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|private[_-]?key|client[_-]?secret|auth(?:[_-]?token)?|credential|cookie|session[_-]?id)"
    t = sub("named secret", r"(?i)([\w.-]*" + name + r"[\w.-]*[\"']?\s*[:=]\s*[\"']?)([^\s\"',;&)}\]]{4,})", t,
            lambda m: m.group(1) + ("REDACTED" if not re.fullmatch(r"(?i)(true|false|null|none|yes|no|\d{1,3}|redacted|\$\{?\w+\}?|<[^>]*>|\*+)", m.group(2)) else m.group(2)))
    t = sub("cli password", r"(?i)((?:sshpass\s+-p|--password(?:=|\s+)|--pass(?:=|\s+)|-pass(?:word)?\s+|IPMI_PASSWORD=|-P\s+(?=\S{6,}\s)(?!\S*\.(?:py|blend|sh|txt)\b)(?!\d+\s))\s*[\"']?)([^\s\"']{3,})", t,
            lambda m: m.group(1) + "REDACTED")
    # generic high-entropy strings (keys, hashes, ids): keep the shape, change the value
    t = sub("long hex", r"\b[0-9a-fA-F]{32,}\b", t, lambda m: fake(m.group(0).lower(), HEX))
    t = sub("high-entropy string", r"(?<![\w/+=-])[A-Za-z0-9+/_-]{40,}={0,2}(?![\w/+=-])", t,
            lambda m: fake(m.group(0), B64) if entropy(m.group(0)) > 4.0 and re.search(r"\d", m.group(0)) and re.search(r"[a-z]", m.group(0)) and re.search(r"[A-Z]", m.group(0)) else m.group(0))
    t = sub("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b", t,
            lambda m: m.group(0) if m.group(0).lower().endswith(("@example.com", "@anthropic.com", "@localhost")) or m.group(0).startswith(("git@", "root@", "noreply@"))
            else "user" + str(int.from_bytes(hmac.new(KEY, m.group(0).lower().encode(), hashlib.sha256).digest()[:2], "big") % 900 + 100) + "@example.com")
    t = sub("public ip", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", t,
            lambda m: m.group(0) if PRIVATE_NET.match(m.group(0)) or any(int(x) > 255 for x in m.group(0).split(".")) or re.fullmatch(r"\d\.\d+\.\d+\.\d+", m.group(0))
            else "203.0.113." + str(hmac.new(KEY, m.group(0).encode(), hashlib.sha256).digest()[0] % 254 + 1))
    return t


def clip(s, n):
    return s if len(s) <= n else s[:n * 2 // 3] + f"\n... [{len(s) - n} characters omitted] ...\n" + s[-n // 3:]


def scrub_obj(o):
    """scrub every string inside a JSON-like value (tool arguments)"""
    if isinstance(o, str):
        return scrub(clip(o, a.tool_chars))
    if isinstance(o, list):
        return [scrub_obj(v) for v in o]
    if isinstance(o, dict):
        return {k: scrub_obj(v) for k, v in o.items()}
    return o


def result_text(c):
    if isinstance(c, str):
        return c
    return "\n".join(b.get("text", "") for b in c or [] if isinstance(b, dict) and b.get("type") == "text")


TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": list(p)[:1]}}}
         for n, d, p in [
    ("Bash", "Run a shell command and return its output.", {"command": {"type": "string"}, "description": {"type": "string"}}),
    ("Read", "Read a file from disk.", {"file_path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}),
    ("Write", "Write a file, replacing it if it exists.", {"file_path": {"type": "string"}, "content": {"type": "string"}}),
    ("Edit", "Replace an exact string in a file.", {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}),
    ("Grep", "Search file contents with a regular expression.", {"pattern": {"type": "string"}, "path": {"type": "string"}}),
    ("Glob", "Find files by glob pattern.", {"pattern": {"type": "string"}, "path": {"type": "string"}}),
    ("WebFetch", "Fetch a URL and answer a question about the page.", {"url": {"type": "string"}, "prompt": {"type": "string"}}),
]]
KNOWN = {t["function"]["name"] for t in TOOLS}
SYSTEM = ("You are a coding and systems agent working in the user's terminal. Use the tools to inspect and change things; "
          "keep the user informed in short plain messages and report results faithfully.")


def load_session(path):
    """-> list of chat messages (assistant entries of one API turn merged), already scrubbed"""
    msgs = []
    for l in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("type") not in ("user", "assistant") or r.get("isSidechain") or r.get("isMeta") or r.get("isApiErrorMessage"):
            continue
        c = r.get("message", {}).get("content")
        if r["type"] == "user":
            if isinstance(c, str):
                if c.strip() and not c.lstrip().startswith(("<command-name>", "<local-command", "<system-reminder>")):
                    msgs.append({"role": "user", "content": scrub(clip(c, a.tool_chars * 2))})
                continue
            for b in c or []:
                if b.get("type") == "tool_result":
                    msgs.append({"role": "tool", "content": scrub(clip(result_text(b.get("content")), a.tool_chars)) or "(no output)"})
                elif b.get("type") == "text" and b.get("text", "").strip() and not b["text"].lstrip().startswith("<system-reminder>"):
                    msgs.append({"role": "user", "content": scrub(clip(b["text"], a.tool_chars * 2))})
        else:
            text, calls = "", []
            for b in c or []:
                if b.get("type") == "text":
                    text += b.get("text", "")
                elif b.get("type") == "tool_use":
                    nm = b.get("name", "tool")
                    args = b.get("input", {})
                    if nm not in KNOWN:            # other tools: keep the action, as a shell-style call
                        args, nm = {"command": nm + " " + json.dumps(args, ensure_ascii=False)}, "Bash"
                    args = scrub_obj(args)
                    calls.append({"type": "function", "function": {"name": nm, "arguments": args}})
            if not text.strip() and not calls:
                continue
            m = {"role": "assistant", "content": scrub(text)}
            if calls:
                m["tool_calls"] = calls
            if msgs and msgs[-1]["role"] == "assistant":       # same turn, split across entries
                msgs[-1]["content"] += m["content"]
                msgs[-1].setdefault("tool_calls", []).extend(m.get("tool_calls", []))
                if not msgs[-1]["tool_calls"]:
                    del msgs[-1]["tool_calls"]
            else:
                msgs.append(m)
    return msgs


def size(m):
    return len(m.get("content") or "") + sum(len(json.dumps(c)) for c in m.get("tool_calls", []))


os.makedirs(a.out, exist_ok=True)
n_tasks = n_sessions = 0
per_project = collections.Counter()
with open(os.path.join(a.out, "tasks-sessions.jsonl"), "w", encoding="utf-8") as out:
    for path in sorted(glob.glob(os.path.join(a.projects, "*", "*.jsonl"))):
        project = os.path.basename(os.path.dirname(path))
        if any(s.lower() in project.lower() for s in a.skip_project):
            continue
        msgs = load_session(path)
        cut_at = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] in ("user", "tool")]
        if not cut_at:
            continue
        n_sessions += 1
        sid = hashlib.sha256(os.path.basename(path).encode()).hexdigest()[:10]
        for i in sorted(random.sample(cut_at, min(a.cuts, len(cut_at)))):
            hist, total = [], 0
            for m in reversed(msgs[:i]):                       # tail of the history within the context budget
                total += size(m)
                if total > a.ctx_chars and hist:
                    break
                hist.append(m)
            hist.reverse()
            while hist and hist[0]["role"] != "user":          # start at a user message
                hist.pop(0)
            if not hist or hist[-1]["role"] == "assistant":
                continue
            out.write(json.dumps({"id": f"sess-{sid}-{i:05d}", "system": SYSTEM, "tools": TOOLS, "thinking": random.random() < 0.3,
                                  "messages": hist[:-1], "turns": [hist[-1]["content"] if hist[-1]["role"] == "user" else hist[-1]]},
                                 ensure_ascii=False) + "\n")
            n_tasks += 1; per_project[project] += 1

with open(os.path.join(a.out, "scrub-report.txt"), "w", encoding="utf-8") as rep:
    rep.write(f"sessions used: {n_sessions}\ntasks written: {n_tasks}\n")
    for p, n in per_project.items():
        rep.write(f"  {p}: {n} tasks\n")
    rep.write(f"extra literals loaded: {len(EXTRA)}\n\nredactions (count, then masked samples; values are never written here):\n")
    for k, n in counts.most_common():
        rep.write(f"\n{k}: {n}\n")
        for s in samples[k]:
            rep.write(f"    ...{s}...\n")
print(open(os.path.join(a.out, "scrub-report.txt"), encoding="utf-8").read())
