#!/bin/sh
# Keep other processes out of benchmark measurements.
#
#   tools/quiet.sh check [SETTLE] [TIMEOUT]
#       Wait until no outside process has been busy for SETTLE seconds (default 180, which is
#       also the cool-down). Exit 1, listing the busy processes, if that has not happened after
#       TIMEOUT seconds (default 900).
#   tools/quiet.sh log FILE [INTERVAL]
#       Append the load average and outside processes using 5%+ CPU to FILE every INTERVAL seconds
#       (default 5) until killed. Run it in the background around the measurement.
#   tools/quiet.sh summary FILE
#       Peak load average and the peak CPU of each outside process seen in a log.
#
# "Outside" means anything other than the benchmark programs listed in OURS. check treats it as
# busy when one process is above PROC_MAX percent CPU (100 = one core) or all outside processes
# together are above TOTAL_MAX; log records every outside process above LOG_MIN. The defaults
# let a lit display (WindowServer ~5-10%) pass and catch a daemon taking a whole core.
# ps reports a decaying average, so a burst shows up for several seconds after it starts.
# check also waits for thermal pressure 0 (nominal; 1 moderate, 2 heavy), and log records it:
# on a laptop the clocks drop ~10% at moderate/heavy, so a warm start skews the result.

OURS=${OURS:-'^(ahbench|gpu_prp|gpu_sq|prst|ps|awk|sh|zsh|bash|sleep|sysctl|date|quiet\.sh)$'}
PROC_MAX=${PROC_MAX:-20}
TOTAL_MAX=${TOTAL_MAX:-50}
LOG_MIN=${LOG_MIN:-5}

# One line per outside process above 1% CPU: "name(pid) cpu%". Last line: "total cpu".
outside() {
    ps -Aco pid=,pcpu=,comm= | awk -v ours="$OURS" '
        { pid = $1; cpu = $2; $1 = ""; $2 = ""; name = substr($0, 3)
          if (name ~ ours) next
          total += cpu
          if (cpu >= 1) printf "%s(%s) %.1f%%\n", name, pid, cpu }
        END { printf "total %.1f\n", total }'
}

busy() {  # print the busy processes, or nothing if quiet
    outside | awk -v pmax="$PROC_MAX" -v tmax="$TOTAL_MAX" '
        /^total / { total = $2; next }
        { lines = lines $0 "\n"; c = $NF; sub(/%$/, "", c); if (c + 0 > pmax) hot = 1 }
        END { if (hot || total > tmax) printf "%sall outside processes: %.1f%%\n", lines, total }'
}

load() { sysctl -n vm.loadavg | tr -d '{}' | awk '{ print $1, $2, $3 }'; }
thermal() { notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{ print $2 }'; }

case $1 in
check)
    settle=${2:-180}; timeout=${3:-900}
    start=$(date +%s); quiet_since=$start; last_report=0
    while :; do
        now=$(date +%s); b=$(busy)
        th=$(thermal); [ "${th:-0}" -gt 0 ] && b="${b:+$b
}thermal pressure $th"
        if [ -n "$b" ]; then
            quiet_since=$now
            if [ $((now - last_report)) -ge 30 ]; then
                echo "$(date '+%H:%M:%S') waiting, load $(load):" >&2
                echo "$b" | sed 's/^/  /' >&2
                last_report=$now
            fi
        elif [ $((now - quiet_since)) -ge "$settle" ]; then
            echo "$(date '+%H:%M:%S') quiet for ${settle}s, load $(load)"
            exit 0
        fi
        if [ $((now - start)) -ge "$timeout" ]; then
            echo "$(date '+%H:%M:%S') still busy after ${timeout}s, load $(load):" >&2
            echo "$b" | sed 's/^/  /' >&2
            exit 1
        fi
        sleep 5
    done ;;
log)
    [ -n "$2" ] || { echo "usage: $0 log FILE [INTERVAL]" >&2; exit 2; }
    interval=${3:-5}
    while :; do
        o=$(outside)
        hot=$(echo "$o" | awk -v lmin="$LOG_MIN" '
            /^total / { next } { c = $NF; sub(/%$/, "", c); if (c + 0 >= lmin) printf " | %s", $0 }')
        ours=$(ps -Aco pcpu=,comm= | awk -v ours="$OURS" '
            { cpu = $1; $1 = ""; if (substr($0, 2) ~ ours && $0 !~ /^ (ps|awk|sh|sleep|date)$/) { n++; t += cpu } }
            END { printf "%.1f%% (%d)", t, n }')
        echo "$(date '+%H:%M:%S') load $(load) outside $(echo "$o" | sed -n 's/^total //p')% ours $ours thermal $(thermal)$hot" >> "$2"
        sleep "$interval"
    done ;;
summary)
    [ -r "$2" ] || { echo "usage: $0 summary FILE" >&2; exit 2; }
    awk -F ' [|] ' '
        { split($1, f, " "); if (f[3] + 0 > peak) { peak = f[3]; at = f[1] }
          t = f[7]; sub(/%$/, "", t); if (t + 0 > tpeak) tpeak = t + 0
          for (i = 2; i <= NF; i++) {
              c = $i; sub(/.* /, "", c); sub(/%$/, "", c)
              name = $i; sub(/\([0-9]+\) [0-9.]+%$/, "", name)
              if (c + 0 > max[name]) max[name] = c + 0; seen[name]++ } }
        END { printf "%d samples, peak 1-min load %s at %s, peak outside total %.1f%%\n", NR, peak, at, tpeak
              for (n in max) printf "  %s: peak %.1f%%, logged in %d samples\n", n, max[n], seen[n] }' "$2" ;;
*)
    sed -n '2,13s/^# \{0,1\}//p' "$0" >&2; exit 2 ;;
esac
