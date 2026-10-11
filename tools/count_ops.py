#!/usr/bin/env python3
"""Count SIMD instructions in the innermost loop of each generated kernel.

usage: tools/count_ops.py build/asm [name-regex]
For each .globl kernel, prints the instruction mix of the loop whose
backward branch closes last-innermost (the label with the most nested
position): fp = f* arithmetic, simd = other vector ALU (trn, ext, shl, and,
add/sub on .2d ...), mov = register moves (likely eliminated at rename),
ld/st = memory ops.
"""
import re
import sys
from pathlib import Path


def classify(ins):
    op = ins.split()[0]
    if op.startswith(("ld", "st", "prfm")):
        return "mem"
    if " v" not in " " + ins.replace(",", " ") and not re.search(r"\bv\d+", ins):
        if re.search(r"\b[dq]\d+\b", ins) and op.startswith("f"):
            return "fp"
        return "int"
    if op in ("mov", "movi", "dup"):
        return "mov"
    if op.startswith("f"):
        return "fp"
    return "simd"


def kernels(text):
    cur, body = None, []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"\.globl\s+(\S+)", s)
        if m:
            if cur:
                yield cur, body
            cur, body = m.group(1), []
            continue
        if cur:
            body.append(s)
    if cur:
        yield cur, body


def innermost(body):
    """Instructions between the innermost numeric label and its b.ne Nb."""
    best = None
    for i, s in enumerate(body):
        m = re.match(r"b\.\w+\s+(\d+)b", s)
        if not m:
            continue
        lab = m.group(1) + ":"
        for j in range(i - 1, -1, -1):
            if body[j] == lab:
                seg = body[j + 1:i]
                if best is None or len(seg) < len(best):
                    best = seg
                break
    return best or []


def main():
    d = Path(sys.argv[1])
    pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
    for f in sorted(d.glob("*.S")):
        for name, body in kernels(f.read_text()):
            if pat and not pat.search(name):
                continue
            seg = [s for s in innermost(body) if s and not s.endswith(":") and not s.startswith(".")]
            c = {}
            for s in seg:
                k = classify(s)
                c[k] = c.get(k, 0) + 1
            print("%-22s fp %3d  simd %3d  mov %3d  mem %3d  int %3d" % (
                name, c.get("fp", 0), c.get("simd", 0), c.get("mov", 0), c.get("mem", 0),
                c.get("int", 0)))


if __name__ == "__main__":
    main()
