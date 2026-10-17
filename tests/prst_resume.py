#!/usr/bin/env python3
"""SIGTERM part way through a Fermat test: kite-prst must stop within a second, save its last
verified state, and finish from there with the residue an uninterrupted run gives.

Both have gone wrong. Up to 0.1.3 the checkpoint was written as "AHNCKPT1" and read as "KITECKP1",
so a test never resumed, and the stop request was read only between Gerbicz rounds, minutes apart
at large n, so a caller waiting a bounded time killed it and lost the lot. In 0.1.4 a stop part way
through a round saved the state at the start of that round, and a round is about a quarter of the
test, so a stop before 25% saved nothing. The stop here lands inside the first round, and the
rerun has to restart within a block or so of where it stopped.

Usage: tests/prst_resume.py PROG    (PROG is build/kite-prst)
"""
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

EXPR = "9357*2^200001-1"
RES64 = "BDD48C45E17CDD83"  # uninterrupted kite-prst run
STOP_AT = 15.0              # percent done before the stop request; the first round ends near 25%
MAX_LOST = 1.0              # percent the restart may fall short of the stop; a block is 0.22%
MAX_STOP_SECONDS = 1.0


def run(prog, cwd, stop_at=None):
    # Progress every half second rather than every 10, so the stop lands part way through a short
    # test on a slow machine as well as a fast one.
    p = subprocess.Popen([prog, EXPR, "-d", "-fermat", "a", "3", "-time", "progress", "0.5"], cwd=cwd,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, buf, stopped, stopped_pct = [], b"", None, None
    while True:
        chunk = p.stdout.read1(4096)
        if not chunk:
            break
        buf += chunk
        *lines, buf = re.split(rb"[\r\n]", buf)
        for line in lines:
            text = line.decode().strip()
            if text:
                out.append(text)
            m = re.search(r"(\d+(?:\.\d+)?)% done", text)
            if stop_at is not None and stopped is None and m and float(m.group(1)) >= stop_at:
                stopped, stopped_pct = time.time(), float(m.group(1))
                p.send_signal(signal.SIGTERM)
    p.wait()
    if buf.strip():
        out.append(buf.decode().strip())
    return p.returncode, out, (time.time() - stopped) if stopped else None, stopped_pct


def main():
    prog = os.path.abspath(sys.argv[1])
    with tempfile.TemporaryDirectory() as d:
        rc, out, took, stopped_pct = run(prog, d, STOP_AT)
        if took is None:
            sys.exit(f"never reached {STOP_AT}% before finishing: {out[-3:]}")
        if rc == 0 or "Terminating..." not in out:
            sys.exit(f"expected a stop, got exit {rc}: {out[-3:]}")
        if took > MAX_STOP_SECONDS:
            sys.exit(f"took {took:.2f}s to stop after SIGTERM, limit {MAX_STOP_SECONDS}s")

        rc, out, _, _ = run(prog, d)
        restart = [l for l in out if "restarting at" in l]
        if not restart:
            sys.exit(f"did not resume from the checkpoint: {out[:3]}")
        restart_pct = float(re.search(r"restarting at ([\d.]+)%", restart[0]).group(1))
        if restart_pct < stopped_pct - MAX_LOST:
            sys.exit(f"stopped at {stopped_pct}% but restarted at {restart_pct}%: the stop dropped unverified work")
        if not any(f"RES64: {RES64}" in l for l in out):
            sys.exit(f"resumed run gave the wrong residue: {out[-2:]}")
    print(f"stopped in {took:.2f}s, {restart[0]}, RES64 {RES64}")


if __name__ == "__main__":
    main()
