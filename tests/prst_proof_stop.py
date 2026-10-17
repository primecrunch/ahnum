#!/usr/bin/env python3
"""SIGTERM while kite-prst compresses its proof, after the test result is printed: it must stop
within a second, and the rerun must print the same result without repeating the test, carry on
from the products already written, and give the raw certificate of an uninterrupted run.

Compression is 9 levels of challenges and powers over 512 points, minutes at n in the millions.
Up to 0.1.7 it read no stop request, so a caller that waits a bounded time killed it after the
result was printed and before the products were written, and the proof was lost.

Usage: tests/prst_proof_stop.py PROG    (PROG is build/kite-prst)
"""
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

EXPR = "9357*2^200001-1"
RES64 = "BDD48C45E17CDD83"     # uninterrupted run
RAW_CERT = "A55E6A55CB989990"  # uninterrupted -proof save 512
MAX_STOP_SECONDS = 1.0


def save(prog, d, stop_when=None):
    """Run -proof save 512 in d; SIGTERM once stop_when(lines, d) is true. Returns the exit
    status, the output lines and how long the stop took."""
    p = subprocess.Popen([prog, EXPR, "-d", "-fermat", "a", "3", "-proof", "save", "512"], cwd=d,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.set_blocking(p.stdout.fileno(), False)
    out, buf, stopped = [], b"", None
    while True:
        chunk = p.stdout.read(4096)
        if chunk:
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]", buf)
            out += [l.decode().strip() for l in lines if l.strip()]
        elif p.poll() is not None:
            break
        else:
            time.sleep(0.005)
        if stop_when and stopped is None and stop_when(out, d):
            stopped = time.time()
            p.send_signal(signal.SIGTERM)
    if buf.strip():
        out.append(buf.decode().strip())
    return p.returncode, out, (time.time() - stopped) if stopped else None


def has_result(out, _d):
    return any(f"RES64: {RES64}" in l for l in out)


def has_products(out, d):
    return has_result(out, d) and any(f.endswith(".prod.3") for f in os.listdir(d))


def stop_and_finish(prog, stop_when, what):
    with tempfile.TemporaryDirectory() as d:
        rc, out, took = save(prog, d, stop_when)
        if took is None:
            sys.exit(f"{what}: finished before the stop: {out[-2:]}")
        if rc == 0 or "Terminating..." not in out:
            sys.exit(f"{what}: expected a stop, got exit {rc}: {out[-3:]}")
        if took > MAX_STOP_SECONDS:
            sys.exit(f"{what}: took {took:.2f}s to stop, limit {MAX_STOP_SECONDS}s")
        products = sorted(f for f in os.listdir(d) if re.search(r"\.prod\.\d+$", f))

        rc, out, _ = save(prog, d)
        if any("% done" in l or "restarting at" in l for l in out):
            sys.exit(f"{what}: the rerun repeated the test: {out[:3]}")
        if not any(f"RES64: {RES64}" in l for l in out):
            sys.exit(f"{what}: the rerun did not print the result again: {out}")
        if rc or not any(f"raw certificate RES64: {RAW_CERT}" in l for l in out):
            sys.exit(f"{what}: exit {rc}, not raw certificate {RAW_CERT}: {out[-2:]}")
        left = [f for f in os.listdir(d) if f.endswith(".done") or f.endswith(".ckpt")]
        if left:
            sys.exit(f"{what}: left behind after the proof: {left}")
        return f"{what}: stopped in {took:.2f}s with {len(products)} products, rerun gave {RAW_CERT}"


def main():
    prog = os.path.abspath(sys.argv[1])
    print(stop_and_finish(prog, has_result, "stop at the result"))
    print(stop_and_finish(prog, has_products, "stop after 4 products"))


if __name__ == "__main__":
    main()
