#!/usr/bin/env python3
"""Measure the max bits/word at which round-off stays below a threshold, for a
grid of FFT lengths and k values, using tools/sweep_err.

Usage: fit_bpw.py BUILD_DIR [threshold] [iters]
Prints one line per (L, k): the largest bpw (step 0.05) with maxerr <= threshold.
"""
import math
import subprocess
import sys

build = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 300


def maxerr(L, k, bpw):
    n = int(bpw * L)
    out = subprocess.run([f"{build}/sweep_err", "neon", str(L), str(k), str(bpw), str(bpw), "1",
                          str(iters)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "maxerr=" in line:
            return float(line.split("maxerr=")[1])
    return 1.0


def limit(L, k):
    # model guess then bisect on a 0.05 grid
    guess = (53 - 6 - 0.5 * math.log2(L) - math.log2(k)) / 2
    lo, hi = guess - 3, guess + 4
    while hi - lo > 0.05:
        mid = (lo + hi) / 2
        if maxerr(L, k, mid) <= thr:
            lo = mid
        else:
            hi = mid
    return lo


Ls = [int(a) for a in sys.argv[4].split(",")] if len(sys.argv) > 4 else \
    [4096, 16384, 65536, 262144, 1048576]
ks = [1, 3, 31, 255, 2047, 9999]
for L in Ls:
    for k in ks:
        b = limit(L, k)
        print(f"L={L} k={k} bpw_max={b:.2f}", flush=True)
