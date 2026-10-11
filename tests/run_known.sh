#!/bin/sh
# Run ahbench prp on every row of known_residues.csv with n <= MAXN.
# Usage: run_known.sh AHBENCH CSV [MAXN] [extra ahbench args...]
bench=$1; csv=$2; maxn=${3:-1000000}; shift 3 2>/dev/null
fail=0; count=0
grep -v '^#' "$csv" | while IFS=, read k n res src; do
    [ "$n" -le "$maxn" ] || continue
    out=$("$bench" prp "$k" "$n" --expect "$res" "$@" 2>/dev/null)
    if [ $? -ne 0 ]; then echo "FAIL $k*2^$n-1 ($src): $out"; echo x >> "${TMPDIR:-/tmp}/ahn_known_fail.$$"
    else echo "ok   $k*2^$n-1  $res"; fi
done
if [ -f "${TMPDIR:-/tmp}/ahn_known_fail.$$" ]; then rm -f "${TMPDIR:-/tmp}/ahn_known_fail.$$"; exit 1; fi
