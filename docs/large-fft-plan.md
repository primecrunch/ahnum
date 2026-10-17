# Plan: kite at production sizes

primecrunch's Fermat work is now n ≈ 2–5M (FFT 160K–400K), and at those sizes kite loses to PRST
on a busy machine. Every speed figure before 2026-10-17 was taken at n ≈ 370K–870K (FFT 28K–64K),
where a test's data and tables fit in L2. This plan moves the work to the sizes primecrunch runs.

**Goal:** with every core busy, kite runs at least 15% more tests per hour than PRST 14.0 at FFT
160K–400K, on both the M4 (4P + 6E) and the M4 Pro (10P + 4E). It is never slower than PRST at
any size on either machine.

## Where it stands (eagle, M4 4P + 6E, 2026-10-17)

ms per squaring, 2007·2^3322001−1 and neighbours, `-t 1`, one test per process, 60 s each:

| Configuration | kite 0.1.8 (256K) | kite, `KITE_NO_NTAB=1` | PRST 14.0 (240K) |
|---|--:|--:|--:|
| 1 test, P-core | 0.708 | 0.787 | 0.706 |
| 4 tests, P-cores | 0.946 | 0.948 | 0.872 |
| 10 tests, all cores | **2.03** | **1.56** | **1.30** |

- One test alone matches PRST, so the gap is not FP work.
- Going from 4 tests to 10 adds 0.7 squarings per ms for kite and 3.1 for PRST. That makes kite
  about 4× slower per E-core than PRST.
- At 10 tests kite gets 5.2 tests per hour at n = 3.32M, against about 8.3 for PRST.
- wolf (M4 Pro, 4 E-cores out of 14) shows a much smaller gap: 1.78 against an estimated 1.71
  ms per squaring from production results.

### Why

Memory traffic per test per squaring at H = 128K complex (FFT 256K), from `neon_init_impl`:

| Array | Size | Notes |
|---|--:|---|
| data (`R × rowstride`) | 2.0 MB | unavoidable; PRST's 240K is 1.9 MB |
| four-step twiddles `t4` | 2.0 MB | the same size as the data, read in every pass-1 group |
| normaliser table `ntab` | 6.0 MB | 48 bytes per complex element; switched on for H ≤ 2^17 |
| weights `A`, `IA`, `mask` | 0.4 MB | |
| total streamed | **≈ 10.4 MB** | PRST: about the data plus small tables |

The M4's E-cluster has 4 MB of L2 shared by 6 cores, so 6 tests there stream about 60 MB a
squaring from SLC and DRAM. The P-cluster's 16 MB holds 4 tests' data but not their tables.

At FFT 49152 the same arrays come to about 3 MB (`t4` 393 KB, `ntab` 1.15 MB). That is why the
2026-10-11 "L1-resident tables" bound (0–2%) and the 2026-10-12 load profile ranked the footprint
last. Both are right at that size and wrong at 256K. The `ntab` cut-off (H ≤ 2^17) was set from
single-test timings, where the table is 11% faster.

## Rules

These are the rules from `speed-improvement-priorities.md`, with one change: every measurement
runs with all cores busy at a production size. A one-core figure is recorded but decides nothing.

- Each step gets an upper bound first: a deliberately wrong build or an existing switch that skips
  the work. A step is dropped if its bound is under 3%.
- Measure on eagle and on wolf, both quiet (`tools/quiet.sh check`), under `caffeinate -i`.
- Report 10 tests on eagle and 14 on wolf, at FFT 128K, 256K and 384K. Run PRST the same way in
  the same session.
- `taskpolicy -b` is not a way to test E-cores: macOS also lowers their clock for background
  work, and PRST printed no progress at all in 60 s under it. Work out the E-cores' share from the
  "P-cores only" and "all cores" runs instead.

## Steps

### 0. Measure production sizes (no code change to the engine)

- `tests/large_fft_cases.csv`: seven primecrunch numbers from n = 2.3M to 5.5M (kite FFT 160K,
  192K, 256K twice, 288K, 384K and 432K), low and high k, each residue confirmed by a passing
  certificate verification.
- `tools/load-bench.sh --prst PATH`: runs the cases as one test, one per P-core and one per core
  for each engine. It reports ms per squaring, tests per hour and kite against PRST as a Markdown
  table. `--ladder` adds FFT 32K–128K for the step 1 cut-off, and `--verify` runs each case's full
  test against its residue. Fold the large-n rows into `tools/kite-bench.sh` once the numbers settle.
  Keep the current small-n rows for the speed history.
- Have `kite time` print the bytes each squaring streams (from the array sizes above), so the
  footprint shows next to every timing.
- Add a row to the README's "kite vs PRST" table at n ≈ 3.3M on both machines. The current table
  is valid only at small n.

Exit: baseline tables for eagle and wolf at 128K, 256K and 384K, for both engines.

### 1. Set the normaliser-table cut-off by footprint (measured: −23% on eagle)

`KITE_NO_NTAB=1` already turns the table off: 2.03 → 1.56 ms at 10 tests on eagle, with no
change at 4 tests and 11% slower alone.

- Replace `H <= (1u << 17)` with a cut-off from the per-test share of L2 with every core busy,
  measured on both machines at FFT 32K, 48K, 64K, 96K and 128K. Expect it near 48K–64K.
- Ship it as 0.1.9 and embed it in the client.

Exit: eagle at 256K, 10 tests: at or below 1.56 ms. Small-n throughput on wolf unchanged (14
tests at 9001·2^662695−1).

### 2. Generate the four-step twiddles instead of storing them (estimate: −15% to −25%)

`t4` holds w_H^(c·k_r) for every element, 2 MB at 256K. It can come from two small tables:
w^(c·k_r) = w^(c·k_hi·S) · w^(c·k_lo) with k_r = k_hi·S + k_lo. That costs one complex multiply
per element in the first pass-1 stage (`dif_t4`), whose kernels run at 80–95% of the FP limit.
It is a cost on P-cores and probably a gain everywhere once tests outnumber the cache.

- Bound first: a wrong-results build that points every row at the same 1-row `t4` slice. That
  removes the 2 MB of traffic and changes no FP work. Measure at 10/14 tests.
- If the bound clears 3%, build the two-table version with an exact table refresh every few
  hundred rows, so round-off stays where `kite_safe_bpw` assumes. Check with `tools/sweep_err.c`
  and the known-residue suite.

Exit: the bound measured, and if it clears the bar, a version within 3% of it.

### 3. A cheaper normaliser without the big table (target: the compressed path ≤ 3% slower alone)

After step 1, large sizes use the compressed path (`A`, `IA` and `mask`, 0.4 MB), which costs
11% on one core. On a busy machine the table path loses anyway, so this only wins back the
single-core cost. That matters on machines where a few tests run alone (P-core-heavy Macs,
`-t` > 1 splits).

- Profile the compressed path with `KITE_PROF=1`, then compare its op count with the table
  path's 576 ops per 128 digits (`tools/count_ops.py`).
- Candidate: precompute only the per-row-pair weight base and do the per-digit 2^(±1/L)
  stepping in registers, like gwnum's compressed weights.

Exit: a single-test gap of 3% or less, or a measured reason it cannot get there.

### 4. Choose the FFT length from loaded cost

`kite_choose_fft_len` picks the cheapest safe length from times taken on one quiet core. For
k = 2007 at n = 3.32M, 240K is safe (13.52 bits per word against a limit of 13.95) and needs
6% less data and work, but 256K measured cheaper alone. Under load, size counts more than
radix mix.

- Measure 240K against 256K, and 320K/336K against 384K, at 10/14 tests. If the smaller length
  wins, weight data size in the cost table, or measure the table with all cores busy.
- Stretch: a lower `BPW_MARGIN` or a better error model, so more k fit the smaller length. That
  needs a round-off sweep (`tools/sweep_err.c`, `tools/fit_bpw.py`) and is not worth doing
  until steps 1–3 land.

Exit: length choice right under load at the three production sizes.

### 5. Threads per test on E-heavy machines

Running fewer, wider tests cuts the bytes in flight: 5 × 2 has half the tests' tables competing
for L2. The client's calibration already measures splits per FFT. After steps 1 and 2, re-run
`kite throughput 2007 3322001 --configs 10x1,5x2,2x5` on eagle and record which split wins.

Exit: the best split known per machine at 256K. No engine change unless a split clearly wins
and `-t 2` scales badly.

### 6. Client safety net (primecrunch, not kite)

The client already calibrates the tests × threads split per FFT. It should calibrate both
engines per FFT the same way and run the faster one, so kite never runs where it loses. That
keeps eagle-like machines on PRST while kite catches up, and moves them over as each step lands.
Changing this is a primecrunch change and needs Andy's go-ahead.

## Expected result

The cumulative figures for eagle at 256K, 10 tests, are estimates except step 1:

| After | kite ms per squaring | against PRST (1.30) |
|---|--:|---|
| 0.1.8 | 2.03 | 56% slower |
| step 1 (measured) | 1.56 | 20% slower |
| step 2 (estimate) | 1.25–1.35 | about level |
| steps 3–4 (estimate) | 1.10–1.20 | 8–18% faster |

Reaching the goal (+15%) on eagle probably needs every step. On wolf, steps 1 and 2 alone should
put kite well ahead, as it already is level there with all the tables in place.
