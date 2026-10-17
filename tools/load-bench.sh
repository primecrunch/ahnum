#!/bin/sh
# Squaring rate with several tests running at once, kite-prst against PRST, at production sizes.
#
#   tools/load-bench.sh [--kite PATH] [--prst PATH] [--secs S] [--counts "1 P N"]
#                       [--cases FILE] [--ladder] [--force]
#   tools/load-bench.sh --verify [--kite-cli PATH] [--cases FILE]
#
# For each case and each test count, both engines run that many single-thread Fermat tests
# (k*2^(n+2i)-1, i = 0..count-1, each in its own directory) for S seconds. The report gives the
# median ms per squaring over every progress sample after each test's first, and the tests per
# hour that rate means at the case's n. Counts are a number, P (one per P-core) or N (one per
# core). The default "1 P N" separates the FP cost (1), the P-cluster's L2 (P) and the E-cluster's
# (N - P), which is where a test's footprint shows. Don't use taskpolicy -b to isolate E-cores:
# macOS lowers their clock for background work.
#
# --ladder adds FFT 32K-128K (k = 2007, about 13 bits per word) to find where kite's tables stop
# fitting. --verify runs each case's full test with `kite prp --expect` instead, one at a time
# (hours at these sizes). Environment variables such as KITE_NO_NTAB=1 reach kite-prst.
#
# The tests are killed by PID, never by pattern, so a crunch client on the same machine is left
# alone. The script refuses to start when the load average is over 1.5 (--force overrides), as
# another program's tests would change every figure.
set -eu

kite=build/kite-prst; prst=; kite_cli=build/kite; secs=60; counts="1 P N"
cases=tests/large_fft_cases.csv; ladder=0; force=0; verify=0
while [ $# -gt 0 ]; do
    case $1 in
        --kite) kite=$2; shift ;;
        --prst) prst=$2; shift ;;
        --kite-cli) kite_cli=$2; shift ;;
        --secs) secs=$2; shift ;;
        --counts) counts=$2; shift ;;
        --cases) cases=$2; shift ;;
        --ladder) ladder=1 ;;
        --force) force=1 ;;
        --verify) verify=1 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

case_list() {
    grep -v '^#' "$cases" | cut -d, -f1-3
    if [ $ladder = 1 ]; then
        for L in 32768 49152 65536 98304 131072; do echo "2007,$((L * 13 + 1)),"; done
    fi
}

if [ $verify = 1 ]; then
    case_list | while IFS=, read -r k n res; do
        [ -n "$res" ] || continue
        start=$(date +%s)
        if "$kite_cli" prp "$k" "$n" --expect "$res" > /dev/null 2>&1; then status=ok; else status=FAIL; fi
        echo "$k*2^$n-1  $status  $(($(date +%s) - start)) s"
    done
    exit 0
fi

[ -x "$kite" ] || { echo "kite-prst not found at $kite (--kite PATH)" >&2; exit 2; }
if [ -n "$prst" ] && [ ! -x "$prst" ]; then echo "PRST not found at $prst" >&2; exit 2; fi
# each test runs in its own directory, so the engines need absolute paths
abs() { (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")"); }
kite=$(abs "$kite")
[ -z "$prst" ] || prst=$(abs "$prst")
load=$(sysctl -n vm.loadavg | awk '{print $2}')
if [ $force = 0 ] && awk -v l="$load" 'BEGIN { exit !(l > 1.5) }'; then
    echo "load average $load: stop other work first, or pass --force" >&2
    exit 1
fi
ncpu=$(sysctl -n hw.ncpu)
pcores=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "$ncpu")
work=$(mktemp -d "${TMPDIR:-/tmp}/load-bench.XXXXXX")
# rate runs in a subshell, so the PIDs it starts go in a file the trap can read
cleanup() {
    [ -s "$work/pids" ] && kill $(cat "$work/pids") 2> /dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT INT TERM

# rate ENGINE K N COUNT: prints "median_ms fft_label"
rate() {
    eng=$1 k=$2 n=$3 count=$4
    run="$work/run"; rm -rf "$run"; mkdir "$run"
    pids=
    : > "$work/pids"
    i=0
    while [ $i -lt "$count" ]; do
        mkdir "$run/$i"
        (cd "$run/$i" && exec "$eng" "$k*2^$((n + 2 * i))-1" -d -fermat a 3 -t 1 -time progress 5 \
            > out 2>&1) &
        pids="$pids $!"
        echo $! >> "$work/pids"
        i=$((i + 1))
    done
    sleep "$secs"
    kill $pids 2> /dev/null || true
    wait 2> /dev/null || true
    : > "$work/pids"
    ms=$(for f in "$run"/*/out; do
        tr '\r' '\n' < "$f" | sed -n 's/.*time per op: \([0-9.]*\) ms.*/\1/p' |
            awk '{ a[NR] = $0 } END { for (i = NR > 1 ? 2 : 1; i <= NR; i++) print a[i] }'
    done | sort -n | awk '{ a[NR] = $1 } END { if (NR) printf "%.3f", a[int((NR + 1) / 2)]; else print "-" }')
    fft=$(grep -h -o 'FFT length [0-9K]*' "$run/0/out" | head -1 | sed 's/FFT length //')
    echo "$ms ${fft:--}"
}

per_hour() { awk -v c="$1" -v ms="$2" -v n="$3" 'BEGIN { if (ms == "-") print "-"; else printf "%.1f", c * 3600 / (ms * n / 1000) }'; }

echo "$(sysctl -n machdep.cpu.brand_string), $pcores P + $((ncpu - pcores)) E, $($kite -v | head -1)${prst:+, $($prst -v | head -1)}, ${secs} s per run${KITE_NO_NTAB:+, KITE_NO_NTAB=$KITE_NO_NTAB}"
echo
if [ -n "$prst" ]; then
    echo "| Case | Tests | kite FFT | kite ms/sq | kite tests/h | PRST FFT | PRST ms/sq | PRST tests/h | kite vs PRST |"
    echo "|---|--:|--:|--:|--:|--:|--:|--:|--:|"
else
    echo "| Case | Tests | kite FFT | kite ms/sq | kite tests/h |"
    echo "|---|--:|--:|--:|--:|"
fi
case_list | while IFS=, read -r k n res; do
    for c in $counts; do
        case $c in P) count=$pcores ;; N) count=$ncpu ;; *) count=$c ;; esac
        set -- $(rate "$kite" "$k" "$n" "$count")
        kms=$1 kfft=$2
        row="| ${k}·2^${n}−1 | $count | $kfft | $kms | $(per_hour "$count" "$kms" "$n") |"
        if [ -n "$prst" ]; then
            set -- $(rate "$prst" "$k" "$n" "$count")
            pms=$1 pfft=$2
            vs=$(awk -v a="$kms" -v b="$pms" 'BEGIN { if (a == "-" || b == "-") print "-"; else printf "%.2f×", b / a }')
            row="$row $pfft | $pms | $(per_hour "$count" "$pms" "$n") | $vs |"
        fi
        echo "$row"
    done
done
