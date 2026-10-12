# Speed improvement priorities

Ranked candidates for making ahnum faster. Each item gets an upper-bound measurement first (a
deliberately wrong build that skips the work), and is dropped if the bound is under ~3% unless
it is nearly free. Measure on a quiet machine (`tools/quiet.sh check`, which also waits for
thermal pressure 0), under `caffeinate -i`. Short runs read ~12% above what this MacBook Pro
sustains, so compare like with like. Results go in `journal.md`; landed gains go in
the README speed history table.

Reference point: 9001·2^662695−1, FFT 49152, 113 µs per squaring, 74.6 s per test.
Time split per squaring: pass-1 inverse stages 18%, normaliser 17%, pass-1 forward stages 16%,
pass-2 forward and inverse stages ~11–12% each, split 10%, row transposes in/out 13%.

## Candidates

The workload that matters is 14 tests running for hours on a machine that is power-limited
(P-cores ~2650 MHz and ~31 W sustained, against ~4.2 GHz for one test alone). Profiled under
that load (2026-10-12), every phase slows by 1.73–1.96× (total 1.87×, 113 → 211 µs per
squaring). FP-bound phases slow least (split 1.73×, p1 forward 1.79×); the normaliser (1.96×),
p1 inverse (1.92×) and pass-2 rows in/out (1.90×) slow most, so their shares grow a little:
p1 inverse 18.6%, normaliser 15.8%, rows in + out 14.2%. The ranking below follows it. 14 tests is
the right count: 12 sustain ~6% less (see Tried and dropped).

1. **Normaliser (80% of its limit, 17% of the time).** After the in-place rewrite it is still
   tied to the carry-chain latency (fmla → frintn per digit). More chains in flight (4 → 8) or
   splitting each group's carries across two independent halves with a fix-up, as pass 1 does
   between threads, would give ~2–3% if it reaches the FP limit. Check first that it is still
   latency-bound at 2650 MHz, where the latency is the same in cycles but the time is longer.

2. **Fewer FP operations: radix-16 stages.** Kernels already run at 80–95% of the FP limit,
   so larger gains need less work, not faster work. At 49152 the pass-2 rows (C = 256) could be
   16 × 16 instead of three radix-8/4 stages, and the pass-1 columns (R = 96) 16 × 6. Fewer
   stages means one less load/store sweep, and fewer twiddle multiplies per point ((r−1)/r per
   stage: 0.23 per bit at radix 16 against 0.29 at radix 8). NEON's 32 registers hold 16 complex
   values at 2 lanes with no room for twiddles, so it needs a 1-lane or split design. Count the
   ops of a prototype kernel with `tools/count_ops.py` before building the rest: the target is
   ≥5% fewer ops in the stages it replaces (~2–3% of a squaring).

3. **Per-test cache footprint, for the E-cluster and under the power cap.** Each test streams
   ~3 MB per squaring from L2, about half of it tables (four-step 393 KB, normaliser 1.15 MB).
   The P-clusters have 16 MB of L2 for 5 tests each, but the E-cluster has 4 MB for 4 tests,
   so E-core tests probably go to the SLC every squaring. The L1-resident-tables bound
   measured 0–2% for 2-minute runs on all 14 tests (2026-10-11). Remeasure it sustained, and
   per E-core test, with the same wrong-results build. If the E-cores gain, the options are
   smaller tables for E-core tests (computed twiddles, compressed weights), which cost FP work
   on P-cores but may not on E-cores.
   Ranked last after the 2026-10-12 load profile: the load-heavy phases slowed 1.90× against
   1.73–1.83× for the FP-bound ones, so memory costs only a little more under load. One cheap
   sustained re-measure of the existing bound before any code.

Below the bar on its own:

- **Pass-2 rows-in stage (~77% of the FP limit).** It loads both rows from L2. Look at
  reordering or a larger first radix so there is more compute per byte loaded. Ceiling ~2 µs
  (~1.7%) at 49152 even at the full FP rate. It is bound by loading the rows, not by FP:
  dropping its twiddle multiplies (56 → 32 ops) did not change its time.

## On hold

- **GPU squaring alongside the CPU tests (prototype in `gpu/`).** `gpu/gpu_prp.m` runs complete
  Gerbicz-checked tests at one length (32768): 173 tests/hour alone. On this MacBook Pro the
  combination loses (2026-10-12, complete tests with `powermetrics`): within a minute of the GPU
  starting, macOS caps the P-cores at 1800 MHz and the CPU at ~5 W while the GPU keeps 15–22 W,
  and with thermal pressure Heavy it leaves P-cores idle. 13 CPU tests + GPU sustain ~244
  tests/hour against ~372 for 14 CPU tests alone. The +22% from 30 s samples was the first
  minute. Worth revisiting only on a desktop (e.g. the Mac mini) with a bigger power budget;
  remaining work there would be other lengths, word size per n, and kernel tuning.

## Done

- In-place normaliser, {osc, 1/base, base·w} records with 4 carry chains: −2.3 to −3.4%
  (`3631fc5`).
- Tangent-form DIT twiddles, radix 2/3/4/8: −2.5 to −4.6% (`3631fc5`).
- Stage order tuned per FFT length: −0.5 to −2% (`3631fc5`).
- Pass-2 row transpose fused into the first/last row-FFT stage: −9% (`3780a4b`).
- Exact fixed-point IBDWT weights, allowing a shorter FFT: −6% (`9dca186`).

## Tried and dropped

- Pass-1 inverse stages (2026-10-12, patch in `docs/experiments/p1-touch-before-inverse.patch`):
  their gap to the forward stages is the L2 load of each group, not the stages' code. With the
  group read into L1 first (correct results), the inverse stages take 18.43 µs against 20.5,
  slightly faster than the forward stages (18.97). The separate read costs 2.9 µs, more than the
  2.1 µs it saves, so the stages already overlap the load better than a pre-touch does (as the
  earlier software prefetch also found). Ceiling ~2.1 µs (1.8%) even if the load were free.

- Fewer concurrent tests under the power cap (2026-10-12, 25-minute runs): 12 tests settle at
  344–347 tests/hour against 366–372 for 14. Each test is faster (124.5 s against 137 s) but not
  by enough. The scheduler moves tests between P- and E-cores, so 10 would not mean P-only; not
  run.

- Tangent-form twiddles in the forward (DIF) stages (2026-10-11, built, patch in
  `docs/experiments/dif-tangent-forward.patch`): −1.5% at 49152, ~0 at 28672 and 65536, +0.8%
  at 114688. The twiddles move into the last forward stages, which had none before (radix 8: 56 →
  68 ops). Pass 2's rows-in stage is load-bound, so dropping its twiddles saves nothing. Pass 1
  needs a second four-step table (~1%). The split scale fold adds 2 ops per iteration.
- Shrinking or computing the four-step and normaliser tables to cut L2 traffic (2026-10-11):
  making both L1-resident (wrong results) saves ~3% on one core, but only 0–2% of 10- and
  14-test throughput, within heat drift. Table traffic is not what slows tests under load, and
  rebuilding tables on the fly costs FP work.
- Removing the k penalty in bits/word (reasoned, not built): wrapped terms carry k⁻¹ mod N in
  the IBDWT with k folded into the base, and gwnum pays the same ~0.5·log2 k bits per word.
- New radices (6, 9, 12, 16) for finer FFT-length steps (reasoned): the chosen length already
  averages 2.66% above the minimum.
- Fusing the normaliser with its neighbouring column stages (2026-10-11): timing pass 1 with the
  normaliser skipped (B) and with the column stages skipped (C) gives B + C − A = 52.5 µs at
  49152, matching pass 2 alone (~53 µs). The joins between phases cost nothing measurable, so
  fusion has no bound. The normaliser costs ~19.5 µs whether it follows the stages or reads cold
  from L2.
- Safety margin 0.1 instead of 0.2 bits/word (2026-10-11): max round-off reaches 0.344–0.375
  within 20,000 squarings (L = 32768 k = 3, L = 49152 k = 255), too close to the 0.40 retry
  threshold for full tests, which run 30–100× more squarings.
- SME radix-8 stages: slower, because handing groups between NEON and SME costs L2 round trips.
- Fusing the real split into the row FFT: under 2%, since the split is FP-bound.
- Finer FFT-length granularity: the chosen length averages 2.66% above the minimum.
- Driver overhead (Gerbicz, setup): 0.75%, nothing to gain.
- Software prefetch of the next pass-1 group: 2% slower.
- Split-kernel doubling folds and moving the special pair out of C: ~0.3% each.
- Compressed weights instead of the table, to cut L2 traffic: slower with 10 tests running.
