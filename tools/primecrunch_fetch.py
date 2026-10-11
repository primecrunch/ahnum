#!/usr/bin/env python3
"""Fetch verified Fermat RES64 residues from the primecrunch API for regression tests.

Uses the public GET /v2/k/{k}/n/{n} (docs/primecrunch-openapi.yaml). The API host is
api.primecrunch.com; the spec's `servers` entry (primecrunch.com) serves the web app.

Only completed `fermat` jobs with a residual are used (LLR residues are a different
quantity). A candidate whose Fermat residues disagree is reported and skipped.

Usage:
  primecrunch_fetch.py K:N [K:N ...]           # explicit candidates
  primecrunch_fetch.py --file pairs.txt        # lines "K N" or "K:N"
  primecrunch_fetch.py --k K --n-from A --n-to B [--max-requests 200]
      # scan an n range; most n are sieved out and return 404, so this is
      # rate-limited and capped
Options:
  --append FILE   append new rows (not already present) to a known_residues.csv
  --rate R        requests per second (default 4)
Output rows: k,n,res64,primecrunch[-imported]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

API = "https://api.primecrunch.com"


def fetch(k, n):
    req = urllib.request.Request(f"{API}/v2/k/{k}/n/{n}",
                                 headers={"Accept": "application/json",
                                          "User-Agent": "ahnum-regression/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def residue_of(doc):
    """Return (res64, imported) or None; raises ValueError on disagreement."""
    res = {}
    for j in doc.get("jobs") or []:
        if j.get("job_type") != "fermat" or j.get("status") != "completed":
            continue
        r = j.get("residual")
        if not r:
            continue
        res[r.upper().rjust(16, "0")] = j.get("imported", False)
    if not res:
        return None
    if len(res) > 1:
        raise ValueError("disagreeing residues: " + ", ".join(res))
    (r, imp), = res.items()
    return r, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--k", type=int)
    ap.add_argument("--n-from", type=int)
    ap.add_argument("--n-to", type=int)
    ap.add_argument("--max-requests", type=int, default=200)
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--append")
    a = ap.parse_args()

    todo = []
    for p in a.pairs:
        k, n = p.replace(":", " ").split()
        todo.append((int(k), int(n)))
    if a.file:
        for line in open(a.file):
            line = line.split("#")[0].strip()
            if line:
                k, n = line.replace(":", " ").replace(",", " ").split()[:2]
                todo.append((int(k), int(n)))
    if a.k is not None:
        todo += [(a.k, n) for n in range(a.n_from, a.n_to + 1)][: a.max_requests]

    have = set()
    if a.append:
        try:
            for line in open(a.append):
                if line.startswith("#") or not line.strip():
                    continue
                k, n = line.split(",")[:2]
                have.add((int(k), int(n)))
        except FileNotFoundError:
            pass

    rows = []
    for k, n in todo:
        t0 = time.time()
        doc = fetch(k, n)
        if doc is not None:
            try:
                got = residue_of(doc)
            except ValueError as e:
                print(f"# skip {k}*2^{n}-1: {e}", file=sys.stderr)
                got = None
            if got:
                r, imp = got
                row = f"{k},{n},{r},primecrunch{'-imported' if imp else ''}"
                print(row)
                if (k, n) not in have:
                    rows.append(row)
                    have.add((k, n))
        dt = time.time() - t0
        if dt < 1.0 / a.rate:
            time.sleep(1.0 / a.rate - dt)

    if a.append and rows:
        with open(a.append, "a") as f:
            for row in rows:
                f.write(row + "\n")
        print(f"# appended {len(rows)} rows to {a.append}", file=sys.stderr)


if __name__ == "__main__":
    main()
