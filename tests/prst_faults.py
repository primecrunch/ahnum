#!/usr/bin/env python3
"""Damaged proof files: kite-prst must reject or mis-verify them the way PRST 14.0.1888 does.

Makes a good proof set for 1203*2^90001-1 (save, then build a certificate), damages copies of
it, and checks each `-proof build` and `-proof cert` against what PRST 14.0.1888 did with the
same files: exit status, error message, certificate RES64 and the files left behind. A damaged
file with a correct .md5 is not an error to either program; it builds or verifies to a
different certificate RES64, which the server catches by comparing RES64s.

Usage: tests/prst_faults.py PROG    (PROG is build/kite-prst, or a prst to check the
expectations themselves against PRST)
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

EXPR, N = "1203*2^90001-1", 1203 * 2**90001 - 1
FP = "prst_1107603830"
GOOD_CERT = "EC81DC6893506908"
POINTS = [f"{FP}.proof.0", f"{FP}.proof.512"] + [f"{FP}.prod.{i}" for i in range(9)]


def fix_md5(path):
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    with open(path + ".md5", "w") as f:
        f.write(digest)


def flip(name, offset, md5=True):  # flip the low bit of one byte
    def damage(d):
        path = os.path.join(d, name)
        with open(path, "r+b") as f:
            f.seek(offset)
            b = f.read(1)[0]
            f.seek(offset)
            f.write(bytes([b ^ 1]))
        if md5:
            fix_md5(path)
    return damage


def set_last_point(x):  # replace the giant in point 512 (16-byte header, then the giant)
    def damage(d):
        path = os.path.join(d, f"{FP}.proof.512")
        with open(path, "rb") as f:
            head = f.read(16)
        words = (x.bit_length() + 31) // 32
        with open(path, "wb") as f:
            f.write(head + words.to_bytes(4, "little") + x.to_bytes(4 * words, "little"))
        fix_md5(path)
    return damage


def remove(name):
    return lambda d: os.remove(os.path.join(d, name))


# (name, damage, expected exit, expected message or certificate RES64, files left)
# The point-512 cases set R = a^(N-1) = x_n / a^2 with a = 3.
BUILD_CASES = [
    ("product bit flipped, stale md5", flip(f"{FP}.prod.3", 100, md5=False), 1,
     f"{FP}.prod.3 is missing or corrupt.", POINTS),
    ("product header fingerprint changed", flip(f"{FP}.prod.3", 8), 1,
     f"{FP}.prod.3 is missing or corrupt.", POINTS),
    ("product missing", remove(f"{FP}.prod.3"), 1,
     f"{FP}.prod.3 is missing or corrupt.", [p for p in POINTS if not p.endswith("prod.3")]),
    ("point 0 is not a^k", flip(f"{FP}.proof.0", 30), 1, "invalid a^k.", POINTS),
    ("R = -1, a root of unity", set_last_point(N - 9), 1, "roots of unity check failed.",
     POINTS),
    ("product bit flipped", flip(f"{FP}.prod.3", 100), 0, "70EC0175CDC0F009", [f"{FP}.cert"]),
    ("point 512 bit flipped", flip(f"{FP}.proof.512", 100), 0, "A7A69BB2C5485D6C",
     [f"{FP}.cert"]),
    ("R = 1, a faked PRP", set_last_point(9), 0, "BBA00A8C81579D44", [f"{FP}.cert"]),
]
CERT_CASES = [
    ("good certificate", lambda d: None, 0, GOOD_CERT),
    ("bit flipped, stale md5", flip("verify.cert", 100, md5=False), 1,
     "Invalid certificate file."),
    ("bit flipped", flip("verify.cert", 100), 0, "B4CEBCB73E13A843"),
    ("iteration changed", flip("verify.cert", 12), 0, "9E8D1D0A63DBEE33"),
    ("no md5 sidecar", remove("verify.cert.md5"), 0, GOOD_CERT),
]


def run(prog, args, d):
    p = subprocess.run([prog, EXPR, "-d"] + args, cwd=d, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).replace("\r", "\n")


def outcome(rc, out, want_rc, want):
    if rc != want_rc:
        return f"exit {rc}, expected {want_rc}"
    if want_rc == 0:
        want = f"certificate RES64: {want}"
    if want not in out:
        return f"no '{want}' in output"
    return None


def main():
    prog = os.path.abspath(sys.argv[1])
    top = tempfile.mkdtemp()
    fail = 0
    try:
        base, cert = os.path.join(top, "base"), os.path.join(top, "cert")
        os.mkdir(base)
        rc, out = run(prog, ["-fermat", "a", "3", "-proof", "save", "512"], base)
        if rc or f"raw certificate RES64: {GOOD_CERT}" not in out:
            print(f"FAIL proof save: exit {rc}\n{out}")
            return 1
        shutil.copytree(base, cert)
        rc, out = run(prog, ["-fermat", "a", "3", "-proof", "build", "512"], cert)
        if rc or f"certificate RES64: {GOOD_CERT}" not in out:
            print(f"FAIL proof build: exit {rc}\n{out}")
            return 1

        for name, damage, want_rc, want, left in BUILD_CASES:
            d = os.path.join(top, "case")
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(base, d)
            damage(d)
            rc, out = run(prog, ["-fermat", "a", "3", "-proof", "build", "512"], d)
            err = outcome(rc, out, want_rc, want)
            files = sorted(f for f in os.listdir(d) if f.startswith(FP) and
                           not f.endswith(".md5"))
            if not err and files != sorted(left):
                err = f"left {' '.join(files)}"
            if err:
                print(f"FAIL build, {name}: {err}\n{out}")
                fail = 1

        for name, damage, want_rc, want in CERT_CASES:
            d = os.path.join(top, "case")
            shutil.rmtree(d, ignore_errors=True)
            os.mkdir(d)
            shutil.copy(os.path.join(cert, f"{FP}.cert"), os.path.join(d, "verify.cert"))
            shutil.copy(os.path.join(cert, f"{FP}.cert.md5"),
                        os.path.join(d, "verify.cert.md5"))
            damage(d)
            rc, out = run(prog, ["-proof", "cert", "verify.cert"], d)
            err = outcome(rc, out, want_rc, want)
            if err:
                print(f"FAIL cert, {name}: {err}\n{out}")
                fail = 1
    finally:
        shutil.rmtree(top)
    if not fail:
        print("prst faults: OK")
    return fail


if __name__ == "__main__":
    sys.exit(main())
