# Speed improvement priorities

Ranked candidates for making ahnum faster. Each item gets an upper-bound measurement first (a
deliberately wrong build that skips the work), and is dropped if the bound is under ~3% unless
it is nearly free. Measure on a quiet machine. Results go in `journal.md`; landed gains go in
the README speed history table.

Reference point: 9001·2^662695−1, FFT 49152, 113 µs per squaring, 74.6 s per test.
Time split per squaring: pass-1 inverse stages 18%, normaliser 17%, pass-1 forward stages 16%,
pass-2 forward and inverse stages ~11–12% each, split 10%, row transposes in/out 13%.

## Candidates

1. **Pass-2 rows-in stage (~77% of the FP limit).** It loads both rows from L2. Look at
   reordering or a larger first radix so there is more compute per byte loaded. Ceiling ~2 µs
   (~1.7%) at 49152 even at the full FP rate, so it is below the bar on its own. It is bound by
   loading the rows, not by FP: dropping its twiddle multiplies (56 → 32 ops) did not change
   its time.

2. **Integer NTT on the GPU, alongside the CPU tests (research, prototype in `gpu/`).** Best
   kernel so far: register-based (16 elements per thread, two exchanges per row) with Shoup
   twiddle multiplies, `AHN_GPU_REG=1 AHN_GPU_SHOUP=1`: 3.83 µs per squaring mod one prime at
   N = 32768 (radix-2 start: 5.55). 9001·2^662695−1 needs ~4 primes at 32768. The CPU loses
   throughput in proportion to GPU busy time (−22% at full load, shared power). Estimated
   CPU + GPU: ~+13% over CPU alone, assuming ~15 µs per squaring for the unbuilt
   prime-combining, carries and reduction. Limits now: the integer complex pipe (`mulhi`), and
   register spills at 16 elements per thread (occupancy 31%). Stockham radix-8 (8 per thread)
   was slower, capped by threadgroup memory.

## Done

- In-place normaliser, {osc, 1/base, base·w} records with 4 carry chains: −2.3 to −3.4%
  (`3631fc5`).
- Tangent-form DIT twiddles, radix 2/3/4/8: −2.5 to −4.6% (`3631fc5`).
- Stage order tuned per FFT length: −0.5 to −2% (`3631fc5`).
- Pass-2 row transpose fused into the first/last row-FFT stage: −9% (`3780a4b`).
- Exact fixed-point IBDWT weights, allowing a shorter FFT: −6% (`9dca186`).

## Tried and dropped

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
