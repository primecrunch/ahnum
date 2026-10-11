# Speed improvement priorities

Ranked candidates for making ahnum faster. Each item gets an upper-bound measurement first (a
deliberately wrong build that skips the work), and is dropped if the bound is under ~3% unless
it is nearly free. Measure on a quiet machine. Results go in `journal.md`; landed gains go in
the README speed history table.

Reference point: 9001·2^662695−1, FFT 49152, 113 µs per squaring, 74.6 s per test.
Time split per squaring: pass-1 inverse stages 18%, normaliser 17%, pass-1 forward stages 16%,
pass-2 forward and inverse stages ~11–12% each, split 10%, row transposes in/out 13%.

## Candidates

1. **Tangent-form twiddles in the forward (DIF) stages.** Shelved below the bar: bound measured
   at 5.3–6.4% (2026-10-11, all forward stage twiddle multiplies removed), expected net ~1.5–3%.
   The output twiddles have to move to the next stage's inputs (p twiddles per butterfly
   instead of p − 1, so radix 8 saves 12 of 28 twiddle ops and radix 4 saves 4 of 12). Pass 2
   also needs a scale fix-up before the split (~1 op per point). The DIT half recovered 45–65%
   of its bound with none of these overheads. Large change.
2. **Pass-2 rows-in stage (~77% of the FP limit).** It loads both rows from L2. Look at
   reordering or a larger first radix so there is more compute per byte loaded. Ceiling ~2 µs
   (~1.7%) at 49152 even at the full FP rate, so it is below the bar on its own.

3. **Integer NTT on the GPU, alongside the CPU tests (research).** Apple GPUs have no FP64, but a
   number-theoretic transform over 32-bit primes (Montgomery multiply, CRT across several
   primes) in Metal could run extra tests while all CPU cores are busy. It is the only idea
   left that could add tens of percent of throughput instead of a few. It has no cheap bound:
   the first step is a prototype NTT-squaring kernel timed against the CPU at 49152.

## Done

- In-place normaliser, {osc, 1/base, base·w} records with 4 carry chains: −2.3 to −3.4%
  (`3631fc5`).
- Tangent-form DIT twiddles, radix 2/3/4/8: −2.5 to −4.6% (`3631fc5`).
- Stage order tuned per FFT length: −0.5 to −2% (`3631fc5`).
- Pass-2 row transpose fused into the first/last row-FFT stage: −9% (`3780a4b`).
- Exact fixed-point IBDWT weights, allowing a shorter FFT: −6% (`9dca186`).

## Tried and dropped

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
