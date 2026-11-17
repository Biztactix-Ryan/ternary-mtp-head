#!/usr/bin/env python3
"""Copy-heavy prompts: edits where most of the answer repeats text from the prompt (the common case in
coding-agent work). Prints per-prompt and overall tok/s and draft acceptance.
usage: bench_copy.py PORT LABEL"""
import json, sys, urllib.request

PY = '''import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("inventory")


@dataclass
class Item:
    sku: str
    name: str
    quantity: int
    unit_price: float

    @property
    def total(self):
        return self.quantity * self.unit_price


def load_items(path):
    items = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                items.append(Item(row["sku"], row["name"], int(row["quantity"]), float(row["unit_price"])))
            except (KeyError, ValueError) as exc:
                log.warning("skipping row %r: %s", row, exc)
    return items


def low_stock(items, threshold):
    return [item for item in items if item.quantity < threshold]


def summarise(items):
    total_value = sum(item.total for item in items)
    total_units = sum(item.quantity for item in items)
    return {"lines": len(items), "units": total_units, "value": round(total_value, 2)}


def write_report(items, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sku", "name", "quantity", "unit_price", "total"])
        for item in sorted(items, key=lambda i: i.sku):
            writer.writerow([item.sku, item.name, item.quantity, item.unit_price, round(item.total, 2)])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarise an inventory CSV")
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    items = load_items(args.source)
    if not items:
        log.error("no usable rows in %s", args.source)
        return 1
    summary = summarise(items)
    log.info("%(lines)d lines, %(units)d units, value %(value).2f", summary)
    for item in low_stock(items, args.threshold):
        log.info("low stock: %s (%s) has %d left", item.sku, item.name, item.quantity)
    if args.report:
        write_report(items, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

NGINX = '''user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    sendfile on;
    tcp_nopush on;
    keepalive_timeout 65;
    server_tokens off;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" "$http_user_agent"';
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    gzip on;
    gzip_types text/plain text/css application/json application/javascript;

    upstream app_backend {
        server 127.0.0.1:8001;
        server 127.0.0.1:8002;
        keepalive 16;
    }

    server {
        listen 80;
        server_name app.example.com;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        server_name app.example.com;

        ssl_certificate /etc/ssl/certs/app.example.com.pem;
        ssl_certificate_key /etc/ssl/private/app.example.com.key;
        ssl_protocols TLSv1.2 TLSv1.3;

        client_max_body_size 10m;

        location /static/ {
            alias /srv/app/static/;
            expires 7d;
        }

        location /api/ {
            proxy_pass http://app_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 30s;
        }

        location / {
            root /srv/app/public;
            try_files $uri /index.html;
        }
    }
}
'''

DOC = '''# Backup runbook

## Scope

This runbook covers the nightly backup of the application database and the uploaded files. It does not
cover the reporting warehouse, which is rebuilt from source every morning.

## Schedule

The backup starts at 01:30 local time and normally finishes within forty minutes. A second job at 03:00
copies the result to the off-site bucket. Both jobs write a line to the operations channel when they end.

## Steps the job performs

1. Take a consistent snapshot of the database volume.
2. Dump the snapshot to a compressed archive and record its checksum.
3. Archive the uploads directory, skipping the cache and temporary folders.
4. Remove local archives older than fourteen days.
5. Copy the new archives and their checksums off site.

## When it fails

Check the job log first. The three common causes are a full backup disk, an expired credential for the
off-site bucket, and a long-running report holding a lock on the database. Free space or renew the
credential, then run the job by hand. If a lock is the cause, wait for the report to finish; do not
kill it without telling the reporting team.

## Restoring

Restore to the staging host first, never straight to production. Verify the checksum before unpacking.
After the restore, run the consistency check and compare the row counts with the figures in the last
successful backup message. Only then schedule the production restore with the on-call lead.

## Contacts

The on-call lead owns the decision to restore. The platform team owns the job and this document.
'''

PROMPTS = [
    "Add type hints to every function and method in this file. Return the complete file and nothing else.\n\n```python\n" + PY + "```",
    "In this file, rename `low_stock` to `below_threshold` everywhere and make the default threshold 25. Return the complete file.\n\n```python\n" + PY + "```",
    "Change this nginx config so the API location has a 60s read timeout and uploads up to 50m are allowed. Return the complete config.\n\n```nginx\n" + NGINX + "```",
    "Add a third backend on port 8003 and enable HTTP/2 on the TLS server. Return the complete config.\n\n```nginx\n" + NGINX + "```",
    "Edit this document: the backup now starts at 00:45 and local archives are kept for thirty days. Return the complete document with only those changes.\n\n" + DOC,
    "Fix any spelling or grammar mistakes in this document and return it in full. If there are none, return it unchanged.\n\n" + DOC,
]

port = sys.argv[1] if len(sys.argv) > 1 else "8097"
label = sys.argv[2] if len(sys.argv) > 2 else ""
tn = tt = dn = da = 0
for i, p in enumerate(PROMPTS, 1):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 900, "temperature": 0, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=900)).get("timings", {})
    n, ms = t.get("predicted_n", 0), t.get("predicted_ms", 0.0)
    tn += n; tt += ms; dn += t.get("draft_n") or 0; da += t.get("draft_n_accepted") or 0
    acc = f"{t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else "n/a"
    print(f"{label} c{i:02d}: {t.get('predicted_per_second', 0):6.1f} tok/s  n={n}  accepted {acc}", flush=True)
print(f"{label} COPY-TOTAL: {1000 * tn / max(tt, 1):.1f} tok/s over {tn} tokens, drafts accepted "
      + (f"{da}/{dn} ({100 * da / dn:.1f}%)" if dn else "n/a"), flush=True)
