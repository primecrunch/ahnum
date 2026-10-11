#!/bin/sh
# Run the reference benchmarks: full PRP test, residue check, time vs reference.
# Usage: tools/bench.sh [AHBENCH] [CSV] [extra ahbench args...]
bench=${1:-build/ahbench}; csv=${2:-tests/benchmarks.csv}; shift 2 2>/dev/null
grep -v '^#' "$csv" | while IFS=, read k n res ref desc; do
    out=$("$bench" prp "$k" "$n" --expect "$res" "$@" 2>/dev/null)
    rc=$?
    secs=$(echo "$out" | sed -n 's/.*FFT [0-9]*  \([0-9.]*\) s.*/\1/p' | head -1)
    status=$([ $rc -eq 0 ] && echo ok || echo FAIL)
    echo "$k*2^$n-1  $status  ${secs}s  reference ${ref}s ($desc)  speedup $(echo "scale=2; $ref / $secs" | bc)x"
done
