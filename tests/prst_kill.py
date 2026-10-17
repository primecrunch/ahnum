#!/usr/bin/env python3
"""SIGKILL part way through the first Gerbicz round: the test must resume from the unverified
checkpoint, not from zero, and finish with the residue and proof of an uninterrupted run. With a
fault before the save, the resume must catch it at the next check and go back to the verified
state.

A stop by SIGTERM verifies and saves the block it reached; SIGKILL, a crash or a power cut gives
no chance to. Up to 0.1.5 the only file was the verified one, written at round ends, a quarter of
the test apart, so a kill lost up to a quarter of the test (25 minutes at n = 3.3M).

Usage: tests/prst_kill.py KITE_PRST KITE    (build/kite-prst and build/kite)
"""
import os
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time

K, N = 9357, 200001
EXPR = f"{K}*2^{N}-1"
RES64 = "BDD48C45E17CDD83"  # uninterrupted run
RAW_CERT = "A55E6A55CB989990"  # uninterrupted -proof save 512
ROUND_END = 447 * 111       # first Gerbicz check, B*M at this n: 24.8%
KILL_FROM = N // 20         # kill once the unverified file is past 5%...
FAULT_AT = 5000             # ...and so past this fault


def ckpt_iter(path):
    try:
        with open(path, "rb") as f:
            head = f.read(56)
    except FileNotFoundError:
        return None
    if len(head) < 56:
        return None
    _k, _n, _base, it, _nl, _fft = struct.unpack("<6Q", head[8:])
    return it


def kill_inside_first_round(cmd, cwd):
    """Start cmd, SIGKILL it once the unverified file (*.u.ckpt) is past KILL_FROM, and check
    nothing had been verified (the matching *.ckpt) by then."""
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    at = unverified = None
    while p.poll() is None:
        unverified = next((f for f in os.listdir(cwd) if f.endswith(".u.ckpt")), None)
        it = ckpt_iter(os.path.join(cwd, unverified)) if unverified else None
        if it is not None and it >= KILL_FROM:
            p.send_signal(signal.SIGKILL)
            p.wait()
            at = it
            break
        time.sleep(0.01)
    if at is None:
        sys.exit(f"{cmd[0]} finished before an unverified save past {KILL_FROM}")
    if at >= ROUND_END:
        sys.exit(f"killed at {at}, after the first check at {ROUND_END}: nothing tested")
    v = ckpt_iter(os.path.join(cwd, unverified.replace(".u.ckpt", ".ckpt")))
    if v:
        sys.exit(f"verified checkpoint at {v} before the first check")
    return at


def text(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True)
    return p.returncode, re.split(r"[\r\n]", p.stdout.decode())


def killed_kite_prst(prst):
    with tempfile.TemporaryDirectory() as d:
        save = [prst, EXPR, "-d", "-fermat", "a", "3", "-proof", "save", "512",
                "-time", "write", "0.2", "progress", "0.5"]
        at = kill_inside_first_round(save, d)
        rc, out = text(save, d)
        restart = [l for l in out if "restarting at" in l]
        if not restart:
            sys.exit(f"kite-prst killed at {at} did not resume: {[l for l in out if l][:3]}")
        pct = float(re.search(r"restarting at ([\d.]+)%", restart[0]).group(1))
        if pct < 100.0 * at / N - 0.1:
            sys.exit(f"killed at {100.0 * at / N:.1f}% but restarted at {pct}%")
        if rc or not any(f"RES64: {RES64}" in l for l in out):
            sys.exit(f"resumed run gave exit {rc}, not RES64 {RES64}: {[l for l in out if l][-3:]}")
        if not any(f"raw certificate RES64: {RAW_CERT}" in l for l in out):
            sys.exit(f"resumed run's proof differs from an uninterrupted one: {[l for l in out if l][-2:]}")
        left = [f for f in os.listdir(d) if f.endswith(".ckpt")]
        if left:
            sys.exit(f"checkpoints left after the test finished: {left}")
        return f"kite-prst killed at {100.0 * at / N:.1f}%, {restart[0].strip()}, RES64 and proof match"


def killed_after_fault(kite):
    with tempfile.TemporaryDirectory() as d:
        args = [kite, "prp", str(K), str(N), "--ckpt", "t.ckpt", "--uckpt", "t.u.ckpt",
                "--uckpt-secs", "0.2"]
        at = kill_inside_first_round(args + ["--fault", str(FAULT_AT)], d)
        rc, out = text(args + ["--expect", RES64], d)
        line = next((l for l in out if "RES64" in l), "")
        if rc or not any(l.startswith("MATCH") for l in out):
            sys.exit(f"resume after a fault at {FAULT_AT} gave the wrong residue: {line}")
        m = re.search(r"gerbicz_errors (\d+)", line)
        if not m or int(m.group(1)) < 1 or "(resumed)" not in line:
            sys.exit(f"expected a resume whose first check fails: {line}")
        return f"faulted at {FAULT_AT}, killed at {at}: check failed on resume, RES64 {RES64}"


def main():
    prst, kite = (os.path.abspath(a) for a in sys.argv[1:3])
    print(killed_kite_prst(prst))
    print(killed_after_fault(kite))


if __name__ == "__main__":
    main()
