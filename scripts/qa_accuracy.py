#!/usr/bin/env python3
"""Short factual questions with answers that can be checked automatically: 60 general knowledge and
60 IT support. Greedy, thinking off. Grading is a keyword / pattern match on the answer, so it is a
coarse accuracy check, and the misses are printed for a manual look.

  qa_accuracy.py PORT LABEL [--out DIR]
  qa_accuracy.py compare DIR LABEL_A LABEL_B
"""
import json, os, re, sys
import requests

# (question, pattern that a correct answer must match), patterns are case-insensitive regular expressions;
# a list means every pattern must match
GK = [
    ("What is the capital of Australia?", r"canberra"),
    ("What is the capital of Canada?", r"ottawa"),
    ("What is the capital of Japan?", r"tokyo"),
    ("What is the capital of Brazil?", r"bras[ií]lia"),
    ("What is the capital of New Zealand?", r"wellington"),
    ("What is the capital of France?", r"paris"),
    ("Which is the largest planet in the Solar System?", r"jupiter"),
    ("Which planet is closest to the Sun?", r"mercury"),
    ("Which planet is known as the Red Planet?", r"mars"),
    ("What is the chemical symbol for gold?", r"\bau\b"),
    ("What is the chemical symbol for sodium?", r"\bna\b"),
    ("Which element has the chemical symbol Fe?", r"iron"),
    ("Which chemical element has atomic number 1?", r"hydrogen"),
    ("What is the chemical formula of water?", r"h[2₂]o"),
    ("At what temperature in degrees Celsius does water boil at sea level?", r"\b100\b"),
    ("At what temperature in degrees Fahrenheit does water freeze?", r"\b32\b"),
    ("Roughly what is the speed of light in a vacuum, in kilometres per second?", r"299\s?79|300\s?000"),
    ("How many continents are there?", r"\b7\b|seven"),
    ("Which is the largest ocean on Earth?", r"pacific"),
    ("Which river flows through Cairo?", r"nile"),
    ("What is the tallest mountain on Earth above sea level?", r"everest"),
    ("Who wrote the play Romeo and Juliet?", r"shakespeare"),
    ("Who painted the Mona Lisa?", r"leonardo|da vinci"),
    ("Who wrote the novel Nineteen Eighty-Four?", r"orwell"),
    ("In what year did the Second World War end?", r"\b1945\b"),
    ("In what year did the First World War begin?", r"\b1914\b"),
    ("In what year did humans first land on the Moon?", r"\b1969\b"),
    ("Who was the first person to walk on the Moon?", r"armstrong"),
    ("In what year did the Berlin Wall fall?", r"\b1989\b"),
    ("In what year did the Titanic sink?", r"\b1912\b"),
    ("Who developed the general theory of relativity?", r"einstein"),
    ("Who discovered penicillin?", r"fleming"),
    ("What is the hardest known natural material?", r"diamond"),
    ("Which gas do plants absorb from the air for photosynthesis?", r"carbon dioxide|co[2₂]"),
    ("Which gas makes up most of Earth's atmosphere?", r"nitrogen"),
    ("How many bones are in the adult human body?", r"\b206\b"),
    ("What is the largest organ of the human body?", r"skin"),
    ("How many chromosomes does a typical human body cell have?", r"\b46\b"),
    ("Which part of the cell is known as its powerhouse?", r"mitochondri"),
    ("What does DNA stand for?", r"deoxyribonucleic"),
    ("What is the largest animal on Earth?", r"blue whale"),
    ("What is the square root of 144?", r"\b12\b|twelve"),
    ("What is 15 multiplied by 12?", r"\b180\b"),
    ("What is 2 to the power of 10?", r"\b1024\b"),
    ("What is the value of pi to two decimal places?", r"3\.14"),
    ("What do the interior angles of a triangle add up to, in degrees?", r"\b180\b"),
    ("How many sides does a hexagon have?", r"\b6\b|six"),
    ("How many minutes are there in a day?", r"\b1440\b"),
    ("How many days are there in a leap year?", r"\b366\b"),
    ("About how many kilometres are in one mile?", r"1\.6"),
    ("How many bits are in a byte?", r"\b8\b|eight"),
    ("What is the decimal number 10 in binary?", r"\b1010\b"),
    ("What is the decimal number 255 in hexadecimal?", r"\bff\b|0xff"),
    ("What is the smallest prime number?", r"\b2\b|\btwo\b"),
    ("What is the currency of Japan?", r"\byen\b"),
    ("Which is the largest country in the world by area?", r"russia"),
    ("Which language has the most native speakers?", r"mandarin|chinese"),
    ("How many players does a football (soccer) team have on the field?", r"\b11\b|eleven"),
    ("What are the three primary colours of light?", [r"red", r"green", r"blue"]),
    ("At what temperature in degrees Celsius does water freeze?", r"\b0\b|zero"),
]
IT = [
    ("What is the default TCP port for SSH?", r"\b22\b"),
    ("What is the default port for Remote Desktop Protocol (RDP)?", r"\b3389\b"),
    ("What is the default port for HTTPS?", r"\b443\b"),
    ("What is the default port for HTTP?", r"\b80\b"),
    ("Which port does DNS use?", r"\b53\b"),
    ("Which port is used for SMTP mail submission with STARTTLS?", r"\b587\b"),
    ("What is the default port for LDAPS?", r"\b636\b"),
    ("Which TCP port does SMB file sharing use on modern Windows?", r"\b445\b"),
    ("Which UDP port does NTP use?", r"\b123\b"),
    ("What is the default port for MySQL?", r"\b3306\b"),
    ("What is the default port for PostgreSQL?", r"\b5432\b"),
    ("What is the default port for Microsoft SQL Server?", r"\b1433\b"),
    ("Which port does Kerberos use?", r"\b88\b"),
    ("Port 993 is the standard port for which secure mail protocol?", r"imap"),
    ("What Windows command clears the DNS resolver cache?", r"ipconfig\s*/flushdns"),
    ("What Windows command forces an immediate Group Policy refresh?", r"gpupdate\s*/force"),
    ("What Windows command scans and repairs protected system files?", r"sfc\s*/scannow"),
    ("What Windows command shows the network hops to a destination?", r"tracert"),
    ("Which Windows tool checks a disk for file system errors?", r"chkdsk"),
    ("What Linux command shows free disk space in human-readable units?", r"df\s+-\w*h"),
    ("What command restarts a systemd service called nginx?", r"systemctl\s+restart\s+nginx"),
    ("Which command shows the logs of a systemd unit?", r"journalctl"),
    ("What command makes a script file executable on Linux?", r"chmod"),
    ("What is the subnet mask for a /24 network?", r"255\.255\.255\.0"),
    ("How many usable host addresses are in a /26 IPv4 subnet?", r"\b62\b"),
    ("How many usable host addresses are in a /30 IPv4 subnet?", r"\b2\b|\btwo\b"),
    ("A Windows PC that cannot reach a DHCP server gives itself an address starting with which two octets?", r"169\.254"),
    ("Which protocol automatically assigns IP addresses to devices on a network?", r"dhcp"),
    ("What are the four steps of the DHCP lease process, in order?", [r"discover", r"offer", r"request", r"ack"]),
    ("Which DNS record type identifies the mail servers for a domain?", r"\bmx\b"),
    ("Which DNS record type maps a hostname to an IPv6 address?", r"aaaa"),
    ("Which DNS record type is used to publish an SPF policy?", r"\btxt\b"),
    ("At which OSI layer does a router operate?", r"layer\s*3|network layer|\b3\b|third"),
    ("At which OSI layer does a traditional Ethernet switch operate?", r"layer\s*2|data.link|\b2\b|second"),
    ("What are the three steps of the TCP three-way handshake?", [r"syn", r"syn.?ack", r"ack"]),
    ("Which HTTP status code means Not Found?", r"\b404\b"),
    ("Which HTTP status code means Forbidden?", r"\b403\b"),
    ("Which HTTP status code indicates a permanent redirect?", r"\b301\b|\b308\b"),
    ("What is the minimum number of disks for RAID 5?", r"\b3\b|three"),
    ("What is the minimum number of disks for RAID 6?", r"\b4\b|four"),
    ("What is the minimum number of disks for RAID 10?", r"\b4\b|four"),
    ("Which RAID level mirrors data across two disks?", r"raid[\s-]*1\b"),
    ("Which IEEE standard defines VLAN tagging?", r"802\.1q"),
    ("What is the IPv4 loopback address?", r"127\.0\.0\.1"),
    ("Which private IPv4 range is most common on home routers, by its first two octets?", r"192\.168"),
    ("Which Windows Security event ID records a successful logon?", r"\b4624\b"),
    ("Which Windows Security event ID records a failed logon?", r"\b4625\b"),
    ("How many FSMO roles are there in Active Directory?", r"\b5\b|five"),
    ("What command-line tool looks up the DNS records of a hostname?", r"nslookup|\bdig\b|resolve-dnsname"),
    ("What ZFS command starts a scrub of a pool named tank?", r"zpool\s+scrub\s+tank"),
    ("What Proxmox command lists the virtual machines on a node?", r"qm\s+list"),
    ("Which PowerShell cmdlet lists running processes?", r"get-process"),
    ("Which PowerShell cmdlet restarts a Windows service?", r"restart-service"),
    ("Which file on Linux holds static hostname to IP address mappings?", r"/etc/hosts"),
    ("What is the full path of the hosts file on Windows?", r"drivers\\+etc\\+hosts"),
    ("What does DHCP stand for?", r"dynamic host configuration protocol"),
    ("What does VPN stand for?", r"virtual private network"),
    ("Which git command shows the commit history?", r"git\s+log"),
    ("Which docker command lists running containers?", r"docker\s+(ps|container\s+ls)"),
    ("Which protocol and port does a ping use?", r"icmp"),
]
SYSTEM = "Answer the question directly in one or two sentences."


def correct(answer, pat):
    a = answer.lower().replace(",", "")
    return all(re.search(p, a) for p in (pat if isinstance(pat, list) else [pat]))


if len(sys.argv) > 1 and sys.argv[1] == "compare":
    d, la, lb = sys.argv[2:5]
    A = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(d, la, "qa.jsonl"), encoding="utf-8")}
    B = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(d, lb, "qa.jsonl"), encoding="utf-8")}
    same = sum(A[i]["content"] == B[i]["content"] for i in A)
    flips = [i for i in A if A[i]["ok"] != B[i]["ok"]]
    print(f"{la} vs {lb}: identical answer text {same}/{len(A)}, graded differently on {len(flips)} {flips}")
    sys.exit(0)

port, label = sys.argv[1], sys.argv[2]
out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.expanduser("~/ttf/data/validate")
os.makedirs(os.path.join(out_dir, label), exist_ok=True)
out = open(os.path.join(out_dir, label, "qa.jsonl"), "w", encoding="utf-8")
res = {}
for name, qs in (("gk", GK), ("it", IT)):
    n_ok, misses, tn, tt = 0, [], 0, 0.0
    for i, (q, pat) in enumerate(qs, 1):
        r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            "max_tokens": 150, "temperature": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600).json()
        ans = r["choices"][0]["message"]["content"]
        ok = correct(ans, pat)
        n_ok += ok; tn += r["timings"].get("predicted_n", 0); tt += r["timings"].get("predicted_ms", 0.0)
        if not ok:
            misses.append((f"{name}{i:02d}", q, ans.strip().replace("\n", " ")[:160]))
        out.write(json.dumps({"id": f"{name}{i:02d}", "q": q, "content": ans, "ok": bool(ok)}, ensure_ascii=False) + "\n")
    res[name] = (n_ok, len(qs), 1000 * tn / max(tt, 1), misses)
print(f"{label:<14} general knowledge {res['gk'][0]}/{res['gk'][1]}  IT support {res['it'][0]}/{res['it'][1]}  "
      f"({res['gk'][2]:.1f} and {res['it'][2]:.1f} tok/s)", flush=True)
for name in ("gk", "it"):
    for mid, q, ans in res[name][3]:
        print(f"   MISS {label} {mid}: {q} -> {ans}", flush=True)
