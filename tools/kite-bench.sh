#!/bin/sh
# kite against PRST on this Mac: the same Fermat and LLR tests primecrunch runs, one test alone
# and then one test per core, with both programs' times and residues side by side. Writes a
# Markdown report for the primecrunch forum.
#
#   curl -fsSLO https://github.com/primecrunch/kite/releases/latest/download/kite-bench.sh
#   sh kite-bench.sh [PATH_TO_PRST]
#
# The released copy of this script is stamped with its kite version and the SHA-256 of that
# release's tarball (.github/workflows/release.yml); it downloads the tarball, checks it and
# runs the kite-prst inside. A kite-prst next to the script is used instead, for testing a
# local build. PRST 14.0 is downloaded from its GitHub release (patnashev/prst) and checked the
# same way, unless a path to a prst is given. Nothing is installed or left behind: everything happens in a temporary directory that
# is removed at the end. The report names the chip, core counts, memory, macOS version and
# program versions, and how busy the rest of the machine was (load average and CPU % only, no
# process names).

set -u
here=$(cd "$(dirname "$0")" && pwd)
KITE_RELEASE="@KITE_RELEASE@"
KITE_SHA256="@KITE_SHA256@"
KITE_URL=${KITE_URL:-"https://github.com/primecrunch/kite/releases/download/v$KITE_RELEASE/kite-$KITE_RELEASE-macos-arm64.tar.gz"}
PRST_VERSION=14.0  # prst 14.0.1888, the version the primecrunch client runs
PRST_URL=${PRST_URL:-"https://github.com/patnashev/prst/releases/download/v$PRST_VERSION/prst-$PRST_VERSION-mac64.7z"}
PRST_SHA256=72f770d3ac03608b861bcd581458848e56bb2a6e582dd929b56a72dab6ff25ba

# Test numbers. Residues checked against PRST 14.0.1888 and kite on an M4 Pro and an M4.
FERMAT_EXPR=${FERMAT_EXPR:-"9357*2^650001-1"}
FERMAT_RES64=${FERMAT_RES64:-"2AB1256594388021"}
LLR_EXPR=${LLR_EXPR:-"4299*2^522773-1"}  # a primecrunch prime: expected LLR result "prime"
# For testing the script: ALL_CORES overrides the all-cores count, BENCH_GO=1 answers "go",
# SETTLE, QUIET_TIMEOUT, REST and THERMAL_WAIT shorten the waits.

SETTLE=${SETTLE:-60}          # seconds of quiet needed before starting
QUIET_TIMEOUT=${QUIET_TIMEOUT:-300}  # how long one quiet check waits before asking
PROC_MAX=20        # one outside process above this % of a core counts as busy
TOTAL_MAX=50       # all outside processes together above this % counts as busy

say() { printf '%s\n' "$*"; }
die() { printf 'kite-bench: %s\n' "$*" >&2; exit 1; }

# ---- preflight ----------------------------------------------------------------------------

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "this needs an Apple Silicon Mac"

work=$(mktemp -d "${TMPDIR:-/tmp}/kite-bench.XXXXXX") || die "cannot make a work directory"
sampler=""
cleanup() {
    [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
    pkill -f "$work/" 2>/dev/null
    rm -rf "$work"
}
trap cleanup EXIT
trap 'say ""; say "Stopped."; exit 130' INT TERM

# kite: a local build next to the script, or the release this script was published with,
# checked against the SHA-256 stamped in at release time before anything is extracted.
if [ -f "$here/kite-prst" ]; then
    KITE_SRC="$here/kite-prst"
    say "Using the kite-prst next to this script."
else
    case $KITE_RELEASE$KITE_SHA256 in
    *@*) die "this copy is not from a release and there is no kite-prst next to it" ;;
    esac
    say "Downloading kite $KITE_RELEASE..."
    curl -fsSL "$KITE_URL" -o "$work/kite.tar.gz" || die "could not download $KITE_URL"
    got=$(shasum -a 256 "$work/kite.tar.gz" | cut -d' ' -f1)
    [ "$got" = "$KITE_SHA256" ] || die "the download does not match its checksum (got $got); not running it"
    tar -C "$work" -xzf "$work/kite.tar.gz" || die "could not unpack the download"
    KITE_SRC="$work/kite-$KITE_RELEASE-macos-arm64/kite-prst"
fi

# PRST: a path given on the command line, or PRST_VERSION from its GitHub release, checked
# against a fixed SHA-256. The binary inside is the one the primecrunch client ships, so every
# report compares against the same PRST. macOS's tar (libarchive) reads 7-Zip.
PRST_SRC=${1:-${PRST:-}}
if [ -n "$PRST_SRC" ]; then
    [ -x "$PRST_SRC" ] || die "$PRST_SRC is not a program"
else
    say "Downloading PRST $PRST_VERSION..."
    curl -fsSL "$PRST_URL" -o "$work/prst.7z" || die "could not download $PRST_URL"
    got=$(shasum -a 256 "$work/prst.7z" | cut -d' ' -f1)
    [ "$got" = "$PRST_SHA256" ] || die "the PRST download does not match its checksum (got $got); not running it"
    mkdir -p "$work/prst-dl" && tar -C "$work/prst-dl" -xf "$work/prst.7z" prst ||
        die "could not unpack PRST"
    PRST_SRC="$work/prst-dl/prst"
fi

# Our own copies under distinct names, so the busy check can tell them from the client's prst.
cp "$PRST_SRC" "$work/prst-bench" && cp "$KITE_SRC" "$work/kite-bench" || die "cannot copy the programs"
xattr -d com.apple.quarantine "$work/kite-bench" "$work/prst-bench" 2>/dev/null
chmod +x "$work/kite-bench" "$work/prst-bench"  # PRST's 7z stores it without the x bit
"$work/kite-bench" -v >/dev/null 2>&1 || die "kite-prst will not run on this Mac"
PRST="$work/prst-bench"
KITE="$work/kite-bench"
OURS='^(prst-bench|kite-bench|ps|awk|sh|zsh|bash|sleep|sysctl|date|notifyutil|kite-bench\.sh)$'

prst_version=$("$PRST" -v 2>&1 | sed -n 's/.*PRST version \([^ ,]*\).*/\1/p' | head -1)
if [ -z "$prst_version" ]; then
    arch -x86_64 /usr/bin/true 2>/dev/null ||
        die "PRST is an Intel program and needs Rosetta: run  softwareupdate --install-rosetta  and try again"
    die "PRST will not run on this Mac: $("$PRST" -v 2>&1 | head -1)"
fi
kite_version=$("$KITE" -v 2>&1 | sed -n 's/^kite version //p')

chip=$(sysctl -n machdep.cpu.brand_string)
ncpu=${ALL_CORES:-$(sysctl -n hw.ncpu)}
pcores=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "?")
ecores=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
macos=$(sw_vers -productVersion)

say "kite $kite_version against PRST $prst_version on $chip (${pcores}P+${ecores}E, $mem_gb GB), macOS $macos"
say "This takes 20-40 minutes. Leave the machine alone while it runs."
say ""

# ---- how busy the machine is --------------------------------------------------------------

load1() { sysctl -n vm.loadavg | awk '{ print $2 }'; }
thermal() { notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{ print $2 }'; }

# Outside processes above 1% of a core, "name (pid) cpu%", then "total CPU".
outside() {
    ps -Aco pid=,pcpu=,comm= | awk -v ours="$OURS" '
        { pid = $1; cpu = $2; $1 = ""; $2 = ""; name = substr($0, 3)
          if (name ~ ours) next
          total += cpu
          if (cpu >= 1) printf "%s (%s) %.0f%%\n", name, pid, cpu }
        END { printf "total %.0f\n", total }'
}

# What is busy, for the person at the keyboard (never written to the report).
busy() {
    o=$(outside)
    # grouped by name ("prst ×14 1309%"), busiest first, only those above 2%
    b=$(printf '%s\n' "$o" | awk -v pmax="$PROC_MAX" -v tmax="$TOTAL_MAX" '
        /^total / { total = $2; next }
        { c = $NF; sub(/%$/, "", c); name = $0; sub(/ \([0-9]+\) [0-9]+%$/, "", name)
          sum[name] += c; cnt[name]++; if (c + 0 > pmax) hot = 1 }
        END { if (!(hot || total > tmax)) exit
              for (n in sum) if (sum[n] > 2)
                  printf "%d\t  %s%s %.0f%%\n", sum[n], n, (cnt[n] > 1 ? " ×" cnt[n] : ""), sum[n]
              printf "-1\t  all other processes together: %.0f%% of one core\n", total }' |
        sort -t "$(printf '\t')" -k1,1nr | cut -f2-)
    th=$(thermal)
    [ "${th:-0}" -gt 0 ] && b="${b:+$b
}  thermal pressure is raised (level $th); the Mac is warm"
    printf '%s' "$b"
}

hint() {  # name the usual suspects in plain words, among processes using 10%+ of a core
    set -- "$(printf '%s\n' "$1" | awk '{ c = $NF; sub(/%$/, "", c); if (c + 0 >= 10 && !/all other/) print }')"
    case $1 in
    *prst*|*primecrunch*) say "  - the primecrunch client is running tests: pause or quit it for now" ;;
    esac
    case $1 in *mds*|*mdworker*) say "  - Spotlight is indexing: it usually finishes on its own" ;; esac
    case $1 in *backupd*) say "  - a Time Machine backup is running" ;; esac
    case $1 in *Chrome*|*Safari*|*firefox*|*WebContent*) say "  - a web browser is busy: closing tabs helps" ;; esac
}

quiet=yes
wait_quiet() {  # $1 = seconds of quiet needed
    settle=$1
    while :; do
        start=$(date +%s); quiet_since=$start; last_busy=""
        while :; do
            now=$(date +%s); b=$(busy)
            if [ -n "$b" ]; then
                quiet_since=$now
                last_busy=$b
            elif [ $((now - quiet_since)) -ge "$settle" ]; then
                printf '\r%60s\r' ""
                return 0
            fi
            if [ -n "$b" ]; then
                top=$(printf '%s\n' "$b" | head -1 | sed 's/^ *//')
                printf '\r\033[KWaiting for the machine to be quiet; busy now: %s' "$top"
            else
                printf '\r\033[KQuiet for %ss of %ss...' $((now - quiet_since)) "$settle"
            fi
            [ $((now - start)) -ge "$QUIET_TIMEOUT" ] && break
            sleep 5
        done
        say ""
        if [ -n "$b" ]; then
            say "The machine is too busy for a fair benchmark. Using CPU right now:"
        else
            say "The machine has not stayed quiet for ${settle}s. Busy most recently:"
        fi
        say "$last_busy"
        hint "$last_busy"
        say ""
        printf 'Make it quieter and press Enter to check again, type go to run anyway, or q to quit: '
        if [ -n "${BENCH_GO:-}" ]; then answer=go; say go
        else read -r answer < /dev/tty || answer=q; fi
        case $answer in
        go|GO|Go) quiet=no; return 0 ;;
        q|Q) exit 1 ;;
        esac
    done
}

# Background sampler for the report: load average and other processes' CPU, every 5 s.
start_sampler() {
    (
        while :; do
            printf '%s %s\n' "$(load1)" "$(outside | sed -n 's/^total //p')"  # load kept for debugging
            sleep 5
        done
    ) > "$work/samples" 2>/dev/null &
    sampler=$!
}

# ---- the runs -----------------------------------------------------------------------------

# Run COUNT copies of one test at once, each in its own directory with its output in `out`.
run_batch() {  # label prog count args...
    label=$1; prog=$2; count=$3; shift 3
    i=0; pids=""
    while [ $i -lt "$count" ]; do
        d="$work/$label.$i"; mkdir -p "$d"
        (cd "$d" && "$prog" "$@" > out 2>&1) &
        pids="$pids $!"
        i=$((i + 1))
    done
    for p in $pids; do wait "$p"; done  # not a bare wait: the sampler never exits
}

result_of() {  # dir -> "seconds residue" (residue: hex RES64, prime, prp, or ?)
    tr '\r' '\n' < "$1/out" | awk '
        /is a probable prime/ { r = "prp" }
        /is prime!/ { r = "prime" }
        /RES64: [0-9A-F]+/ { match($0, /RES64: [0-9A-F]+/); r = substr($0, RSTART + 7, RLENGTH - 7) }
        /[Tt]ime: [0-9.]+ s/ { match($0, /[Tt]ime: [0-9.]+/); t = substr($0, RSTART + 6, RLENGTH - 6) }
        END { printf "%s %s\n", (t == "" ? "?" : t), (r == "" ? "?" : r) }'
}

# Summarise a batch: "mean_seconds residue_or_MIXED"
summarise() {  # label count
    i=0; : > "$work/$1.res"
    while [ $i -lt "$2" ]; do result_of "$work/$1.$i" >> "$work/$1.res"; i=$((i + 1)); done
    awk '{ s += $1; n++; r[$2]++; last = $2 }
         END { m = 0; for (x in r) m++
               printf "%.1f %s\n", s / n, (m == 1 ? last : "MIXED") }' "$work/$1.res"
}

# 20 s between runs, and up to 2 more minutes for thermal pressure to fall back to nominal. A
# Mac that stays warm (a busy machine, or a laptop in a warm room) is noted in the report.
warm=no
cooldown() {
    sleep "${REST:-20}"
    waited=0; th=$(thermal)
    while [ "${th:-0}" -gt 0 ] && [ $waited -lt "${THERMAL_WAIT:-120}" ]; do
        sleep 10; waited=$((waited + 10)); th=$(thermal)
    done
    [ "${th:-0}" -gt 0 ] && warm=yes
}

rows=""
mismatch=""
add_row() {  # test expr mode count expected prst_label kite_label wall_prst wall_kite
    test=$1; expr=$2; count=$4; want=$5; plabel=$6; klabel=$7; wall_p=$8; wall_k=$9
    set -- $(summarise "$plabel" "$count"); pt=$1; pr=$2
    set -- $(summarise "$klabel" "$count"); kt=$1; kr=$2
    ratio=$(awk -v p="$pt" -v k="$kt" 'BEGIN {
        if (k <= 0 || p <= 0) { print "?"; exit }
        if (p >= k) printf "%.2f× faster", p / k; else printf "%.2f× slower", k / p }')
    if [ "$count" -gt 1 ]; then
        ptime=$(awk -v t="$pt" -v n="$count" -v w="$wall_p" 'BEGIN { printf "%.1f s, %.0f/h", t, n * 3600 / w }')
        ktime=$(awk -v t="$kt" -v n="$count" -v w="$wall_k" 'BEGIN { printf "%.1f s, %.0f/h", t, n * 3600 / w }')
        mode_s="$count tests"
    else
        ptime="$pt s"; ktime="$kt s"; mode_s="1 test"
    fi
    if [ "$pr" = "$kr" ] && [ "$kr" = "$want" ]; then
        res="$kr ✓"
        [ "$count" -gt 1 ] && res="all $kr ✓"
    else
        res="**PRST $pr, kite $kr, expected $want**"
        mismatch="$mismatch
- $test $expr ($mode_s): PRST $pr, kite $kr, expected $want"
    fi
    pretty=$(printf '%s' "$expr" | sed 's/\*/·/; s/-1$/−1/')
    rows="$rows
| $test | $pretty | $mode_s | $ptime | $ktime | $ratio | $res |"
}

timed_batch() {  # label prog count args... ; sets $wall
    t0=$(date +%s)
    run_batch "$@"
    wall=$(( $(date +%s) - t0 ))
    [ $wall -lt 1 ] && wall=1
}

fermat_args="-d -fermat a 3"
step=0
steps=8
progress() { step=$((step + 1)); say "[$step/$steps] $*"; }

wait_quiet "$SETTLE"
say ""
start_sampler
t_start=$(date +%s)
load_start=$(load1)  # before our tests, which the load average would count

for mode in 1 "$ncpu"; do
    for test in Fermat LLR; do
        if [ $test = Fermat ]; then expr=$FERMAT_EXPR; args=$fermat_args; want=$FERMAT_RES64
        else expr=$LLR_EXPR; args="-d"; want=prime; fi
        what="$test, $( [ "$mode" = 1 ] && echo "one test" || echo "$mode tests at once")"
        progress "PRST: $what"
        cooldown
        timed_batch "p.$test.$mode" "$PRST" "$mode" "$expr" $args; wp=$wall
        progress "kite: $what"
        cooldown
        timed_batch "k.$test.$mode" "$KITE" "$mode" "$expr" $args; wk=$wall
        add_row "$test" "$expr" "$mode" "$mode" "$want" "p.$test.$mode" "k.$test.$mode" "$wp" "$wk"
    done
done

kill "$sampler" 2>/dev/null; sampler=""
minutes=$(( ($(date +%s) - t_start + 59) / 60 ))
noise=$(awk -v l0="$load_start" '{ s += $2; if ($2 > po) po = $2; n++ }
    END { printf "load average %s before the runs", l0
          if (n) printf ", other processes %.0f%% of one core on average (peak %.0f%%) during them", s / n, po }' "$work/samples")

# ---- report -------------------------------------------------------------------------------

chip_short=$(printf '%s' "$chip" | sed 's/^Apple //; s/ /-/g')
report="$PWD/kite-bench-$chip_short-$(date +%Y-%m-%d).md"
i=2  # a second run the same day gets -2, -3, ... rather than replacing the first report
while [ -e "$report" ]; do
    report="$PWD/kite-bench-$chip_short-$(date +%Y-%m-%d)-$i.md"; i=$((i + 1))
done
{
    say "### kite vs PRST — $chip (${pcores}P+${ecores}E, $mem_gb GB), macOS $macos"
    say ""
    if [ $quiet = yes ]; then
        say "kite $kite_version · PRST $prst_version (Rosetta) · quiet machine: $noise · $minutes min"
    else
        say "kite $kite_version · PRST $prst_version (Rosetta) · **run on a busy machine, times are not a fair comparison**: $noise · $minutes min"
    fi
    if [ $warm = yes ]; then
        say ""
        say "Thermal pressure stayed raised between runs, so clocks were probably reduced."
    fi
    if [ -n "$mismatch" ]; then
        say ""
        say "**Residue mismatch** (please post this even if nothing else):$mismatch"
    fi
    say ""
    say "| Test | Number | Mode | PRST | kite | kite vs PRST | Result |"
    say "|---|---|---|---|---|---|---|"
    printf '%s\n' "$rows" | sed '/^$/d'
    say ""
    say "Times are each test's own reported time (mean over the tests running at once); /h is tests"
    say "per hour for the whole machine. Fermat is \`-d -fermat a 3\` and LLR is \`-d\`, as the client runs them."
} > "$report"

say ""
cat "$report"
say ""
say "Saved as $report — paste it into the forum as it is."
