# ahnum journal

Times are local (2026-10-11). Entries before 12:30 were reconstructed from file timestamps and
process logs after CLAUDE.md asked for this journal, so they are approximate to a few minutes.

## 2026-10-11 10:30 — Plan

Goal: a library for squaring modulo Riesel numbers k·2^n−1, as fast as possible on Apple
Silicon, for Fermat PRP testing. The workload is k < 10000 from n ≈ 400K upward (throughput),
plus record attempts above 69·2^24612729−1.

Decisions from the planning questions:
- CPU first. NEON baseline, with an SME path on M4 where it measures faster.
- The hot code is assembly. C, C++ or Rust were ruled out because the AMD64 state of the art
  (gwnum) is hand-written assembly. Here a Python generator emits AArch64 assembly, and thin C
  handles setup, threads and the API.
- Gerbicz plus round-off checking. Verified residues come from the primecrunch API.
- Throughput matters most. The tests × threads split has to be measured per FFT size.

## 10:48 — Pinning the residue convention

There are several plausible "Fermat residue" definitions: 3^(N−1), 3^(N+1), the half
exponents, and so on. A GMP oracle printing all of them took 38 minutes for one 370K-bit
number. The project's own first FFT finished sooner, and it showed that primecrunch's
`F5FDA01218F20439` for 9357·2^369717−1 is the low 64 bits of 3^(N−1) mod N.

## 10:55 — Scalar reference IBDWT

The generalised IBDWT (Crandall–Fagin, extended to k·2^n±c by Percival 2003) with k folded
into the place values, P_0 = 1 and P_j = k·2^{s_j}. It matched GMP on the first run,
including FFT lengths that are not powers of two.

## 11:05 — The cost of k, and fixing it

Measured round-off showed that this layout loses about 0.8·log2(k) bits per FFT word. For
k ≈ 9357 that is about 10.5 bits per word instead of about 21, roughly doubling the FFT length.
The cause is weights of size k^{1−j/L}: their spread of k inflates the convolution's error.

gwnum's source comments describe a different approach, which I re-derived: fold k into the
irrational base instead (T = n + log2 k, s_j = ⌈jT/L⌉), so the weights stay in [1, 2). The
wrapped terms then come out as integers divided by k. Multiplying every output by k makes
them integers again, at a cost of log2(k) bits of the 53, which is about 0.5·log2 k bits per
word. Squaring then yields k·V², so numbers are stored as x/k ≡ x·2^n (mod N), a
Montgomery-like form with R = 2^n. Bits that land above 2^n are "folded": H·2^n with
H = qk + r becomes q + r·2^n.

Result: k = 9357 went from about 10.5 to about 14.3 bits per word, a FFT about 35% shorter.
A fold bug for k = 1 (no top region, so the code overwrote the last digit) showed up in the
GMP comparison straight away.

## 11:20 — Two-pass structure, prototyped in C

A complex FFT of length H = L/2 on packed pairs, four-step H = R×C:
- **Pass 1:** column FFTs on groups of 8 columns, fused with normalisation.
- **Pass 2:** row FFTs on row pairs (p, p′) with k_r′ = −k_r, so the real-split partners
  (k, H−k) are in the same pair.

Carries cross group boundaries. The first group is held back as integer digits until the last
group's carries, and the wrapped top, are known. It matched GMP on the first run. It was
written as plain C so that the assembly could be checked against an independent path.

## 11:25 — Assembly generator

`gen/gen_kernels.py` emits DIF/DIT stage kernels for radix 2, 3, 4, 5, 7 and 8. Each comes in
three variants: stage twiddles, no twiddles, and a four-step twiddle table at the same offsets
as the data. Complex numbers occupy two NEON registers (re lanes, im lanes), and twiddles use
by-element FMA. All 36 variants matched a scalar reference on the first test.

Pass 2 transposes a row pair so that row p is lane 0 and row p′ is lane 1. Every row-FFT
butterfly then uses the same twiddles in both lanes, and the real-split partner of (p, q) is
lane 1 at C−1−q, so pairing costs one `ext` lane swap.

Timings for 9357·2^369717−1, single core:

| Step | µs per squaring |
|---|---|
| Plain C two-pass | 4537 |
| Assembly FFT stages | 142 |
| + assembly normaliser | 98 |
| + assembly split and transposes | 87 |
| + table-driven, 4-row interleaved normaliser | 69 (L = 28672) |

The normaliser was limited by latency, not throughput. The carry chain went
fadd → fmul → frintn per digit, about 9 cycles. Two changes fixed it: rewriting the chain as
c′ = rint(fma(c, 2^−b, x·2^−b)), which is still exact because everything is a power-of-two
scale, and interleaving two independent row pairs.

## 11:40 — Gerbicz, checkpoints, fault injection

d_t = ∏ x_{jB}, checked as x_0·d_t^(2^B) = d_{t+1} every M blocks, with rollback. The first
fault-injection test changed a frequency-domain value. That showed up as round-off error 0.5
and went through the "larger FFT" path instead of Gerbicz. Injecting +1 into an integer digit
(a valid encoding of a wrong number) exercises the Gerbicz path: it was detected and rolled
back, and the residue came out correct.

## 11:50 — Threading

A fork-join pool. Pass 2 splits row pairs across threads; pass 1 splits column groups, and
each thread holds back its first group. Scaling stopped at 4 threads at first. The workers
spun for a fixed count of `yield`s, ran out and fell asleep between passes, so every pass paid
the wake-up latency. Spinning for 5 ms by the ARM counter fixed it.

Results:
- 69·2^24612729−1: 10.3 ms per squaring on 1 thread, 1.28 ms on 8 threads. 9 threads were
  slower than 8.
- At 256K the FFT stays in L2. 5 threads gave a 4.1× speedup and 10 threads collapsed. With
  every P-core spinning, one preempted thread stalls the whole barrier, and a second cluster
  adds cross-cluster L2 traffic.

## 12:00 — FFT length selection

Fitted from measured round-off over L = 4K … 1M and k = 1 … 9999:
bpw_max ≈ 21.44 − 0.305·(log2 L − 12) − 0.525·log2 k. Round-off is quantised (0.25, 0.3125,
…) because the outputs are near 2^50, where one ulp is 0.25.

The speed of a given length depends heavily on its odd factors. At 28K, 24576 = 2^13·3 runs at
61 µs and 28224 = 2^6·3²·7² at 88 µs, while 32768 is barely slower than 28224 despite 16% more
data. `ahbench tune` measures every length, and selection takes the fastest safe one.

## 12:10 — Kernel ceiling

Per-kernel microbenchmarks show every kernel at about 4 FP instructions per cycle. NEON FMLA
peaks at 35.9 GFLOPS, which implies 4 pipes at 4.49 GHz. Further gains have to come from fewer
instructions, not better scheduling.

## 12:15 — SME measurements

| Operation | Throughput |
|---|---|
| Streaming-SVE FMLA | 15 GFLOPS (39 per cluster) |
| FP64 FMOPA | 492 GFLOPS from one core; 935 with both clusters' units |
| Streaming load/store | about 440 GB/s from L1 to 8 MB |

Streaming-SVE vector arithmetic is slower than NEON. An FFT stage written as 8×8 complex
matrix products, with the twiddles folded into per-j matrices, uses about 5× the flops of a
radix-8 butterfly. It still projects to about 2.2× faster per stage on the one core driving
SME. The unit is shared per cluster.

## 12:20 — Edge cases

- `ahn_prp` hung on 8191·2^25000−1, because N is divisible by 3, so dividing by base² mod N
  has no solution. It now uses CRT: 0 mod the base-prime part of N, and y/base² mod the rest.
- A wrong square at n ≤ 20 with k = 9357 came from a carry that rippled up into the top region
  and was treated as sitting at 2^{s_L}.
- The automatic backend fell back to the scalar reference for tiny n but freed state that had
  never been allocated.

## 12:35 — primecrunch residues

The API is served from `https://api.primecrunch.com`. The OpenAPI `servers` entry,
`https://primecrunch.com`, returns the web app's HTML for `/v2/...`.

GET `/v2/k/{k}/n/{n}` is public and lists jobs with a `residual`. Only `fermat` jobs are used,
because LLR residues are a different quantity. No endpoint lists candidates, so
`tools/primecrunch_fetch.py` takes (k, n) pairs or does a rate-limited scan; sieved-out n
return 404.

From n = 369500…369699, 13 residues for k = 9357 were fetched, and ahbench matched all 13
(10 tests in parallel). The primecrunch record for 9357·2^369717−1 shows the original result
came from a darwin/amd64 client in 30 s. ahnum runs it natively in 26 s on one P-core with
Gerbicz checking.

Benchmark caveat: this machine also runs a distributed compute project, so the throughput
numbers so far come from a loaded system.

## 12:40 — Quiet-machine baselines

The distributed compute project was paused, giving these baselines:

| Test | Configuration | Time per squaring | Throughput / time per test |
|---|---|---|---|
| 9357·2^369717−1 | 1 core | 69 µs | 26 s per test |
| 9357·2^369717−1 | 10 tests × 1 thread | 86 µs each | 1125 tests/hour |
| 9357·2^369717−1 | 14 tests × 1 thread (P- and E-cores) | 110 µs each | 1244 tests/hour |
| 69·2^24612729−1 | 1 thread | 6.5 ms | 45 hours |
| 69·2^24612729−1 | 10 threads | 0.77 ms | 5.3 hours |

The loaded figures earlier (10.3 ms on 1 thread, 1.28 ms on 8 threads) were about 40% worse.

At n ≈ 370K, splitting one test across threads always loses: 5 tests × 2 threads gives 892
tests/hour and 1 test × 10 threads gives 510. One test per core is the right setting there.

## 12:50 — Layout change for SME

Numbers moved from 2-column lane-groups to 8-column blocks of 128 bytes, [re c0..c7][im c0..c7].
For the NEON kernels this only changes load/store offsets. The generator now emits stage kernels
for both layouts: blocks for pass 1, pairs for the pass-2 scratch. Squaring speed did not change
(70 µs at 28672). Now 72 stage-kernel variants are tested.

## 13:00 — SME radix-8 stage kernel

`ahn_sme_r8` replaces each radix-8 batch (8 points × 8 lanes) with Y = M_j X, where M_j is a
complex 8×8 matrix with the twiddles folded in. The DIT direction uses
M = conj(F8)·diag(conj w), so one kernel and two sets of tables cover both directions.

Probe results that shaped the kernel:

| Operation | Cost |
|---|---|
| FMOPA FP64 | latency 4.6 cycles, throughput 1 per 1.15 cycles (needs 4 tiles in flight) |
| smstart + smstop | about 9 ns |
| `st1d` of a ZA tile slice | 7.7 cycles |
| MOVA of 4 rows to Z registers, then normal stores | about 1.2 cycles per 64-byte vector |
| `ld1d` | 0.3–0.7 cycles per vector |

Along the way:
- `smstart` and `smstop` zero every FP register, including the callee-saved d8–d15. A timing
  probe that did not save them produced nonsense clock readings.
- Benchmarking stores into the same 2 KB repeatedly made them look 2× slower than streaming
  stores do.
- I first counted the FMOPA cost wrong: a pair of items needs 64, not 32. With that corrected,
  the kernel is FMOPA-bound at 0.85 cycles/point against 1.3 for NEON. The floor for this
  formulation is 0.58.

## 13:20 — SME in the squaring loop: slower

Pass-1 radix-8 column stages were wired to SME, with the four-step twiddle applied by a small
NEON kernel. The results were correct and slower:

| FFT length | NEON | SME |
|---|---|---|
| 24576 | 64 µs | 83 µs |
| 28672 | 70 µs | 100 µs |
| 262144 | 720 µs | 903 µs |

A hand-off microbenchmark explains it. Alternating an SME stage and a NEON pass over the same
8 KB costs 577 ns, against 180 ns for the two run separately. The SME unit sits at L2, so each
change of owner moves the group's cache lines out of the core's L1 and back, at about 28 cycles
per line. Normalisation has to run on NEON, because streaming-SVE arithmetic is about 5× slower,
so each group changes hands at least twice per squaring.

Throughput cannot gain either: an SME test is slower than a NEON test on the same core, and the
unit is shared by 5 cores.

Decision: keep `--backend sme` as a documented, tested opt-in. NEON stays the default.

The full 9357·2^369717−1 run:
- NEON: 26.7 s
- SME: 35.6 s

Both give RES64 F5FDA01218F20439.

## 13:10 — Benchmark: 8243·2^872226−1 (known prime)

On the quiet machine, NEON, FFT 71680 (12.17 bits/word):

| Setup | Time per squaring | Result |
|---|---|---|
| 1 test, 1 core | 198 µs | 172.6 s; RES64 0000000000000001 (probable prime), max round-off 0.11, no Gerbicz errors |
| 10 tests × 1 thread | 254 µs each | 162 tests/hour |
| 14 tests × 1 thread | 325 µs each | 178 tests/hour |
| 5 tests × 2 threads | — | 149 tests/hour |
| 1 test × 10 threads | 41 µs | 101 tests/hour, but one test takes about 36 s |

The number is a known prime, so it was added to `tests/known_residues.csv` as a PRP check.

## 13:20 — Reference benchmark: 9001·2^662695−1

primecrunch records this known prime's Fermat job at 195 s on wolf (this M4 Pro, darwin/arm64
client); its LLR proof took 407 s. ahnum on one quiet P-core: 91.5 s (138 µs/iter, FFT 51200,
12.94 bits/word, max round-off 0.31, no Gerbicz errors), 2.13× faster than the reference. It
is now the reference benchmark (`tests/benchmarks.csv`, `tools/bench.sh`).

The 195 s comes from the distributed client, which may have been sharing the machine with other
work at the time, so it is an upper bound rather than a quiet single-core figure.

## 13:35 — One test per core on 9001·2^662695−1

Full Gerbicz-checked PRP runs as separate processes on the quiet machine (all residues matched):

| Concurrent tests | Time per test | Throughput |
|---|---|---|
| 1 | 91.6 s | 39 tests/hour |
| 10 (P-cores) | 121 s | 298 tests/hour |
| 14 (P- and E-cores) | 156 s | 321 tests/hour |

The squaring-rate samples (`ahbench throughput`) suggested 342 and 385 tests/hour. The full
runs come out about 15% lower, which is consistent with Gerbicz overhead, context setup and
sustained all-core clocks.

Each test is 32% slower when all P-cores are busy (91.6 s to 121 s), from shared L2 and power.
The E-cores add 8%.

Against the reference client's 195 s per test: if it ran one test per P-core, it would manage
about 185 tests/hour, so ahnum is about 1.6–1.7× on throughput. That assumes the 195 s was
measured under a comparable load, which is not known.

## 14:40 — prst on the same workload, and a weight-precision bug

The client's prst (x86_64 gwnum, translated by Rosetta 2), run exactly as the client does it
(`prst "9001*2^662695-1" -d -fermat a 3`), on the quiet machine:

| Concurrent tests | prst per test | prst throughput | ahnum per test | ahnum throughput |
|---|---|---|---|---|
| 1 | 90.2 s | — | 91.6 s | — |
| 10 | 115.9 s | 307.6/h | 121 s | 297.5/h |
| 14 | 152.6 s | 329.4/h | 156 s | 321/h |

So the 195 s in the primecrunch record was measured under load. Even translated, gwnum matched
ahnum on one core and was about 3% ahead on throughput.

The difference was FFT length: gwnum ran at 48K (49152), ahnum at 51200. At 49152 ahnum's max
round-off reached 0.5 (outright failure), and so did the scalar reference backend. The problem
was therefore in the arithmetic, not the NEON code.

A position histogram of round-off above 0.2 was the clue: there were 10× more large errors in
the top tenth of the words than in the bottom tenth. FFT round-off has no reason to depend on
position; the weights do. The exponent of w_j was computed as (s_j·L − n·j)/L − j·log2(k)/L.
Both terms grow to about log2 k ≈ 13 at the top while their difference stays in [0, 1), so double
precision lost about 4 bits to cancellation. Weights near the top were off by a few ulp, and
weight errors enter the convolution like extra round-off.

The fix:
- Compute log2 k as a 64-bit binary fraction by repeated squaring in fixed point.
- Define T′ = n + log2k_fixed.
- Derive both the digit boundaries s_j and the weight exponents from T′ as exact 128-bit integer
  ratios.

z^L = 2^T′ then differs from k·2^n by a relative 2^-64, which is below double precision.

Results:
- Max round-off at 49152 fell from 0.5 to 0.19 over 300 squarings and 0.22 over 20,000.
- The refitted k penalty is 0.50·log2 k instead of 0.525.
- Large k gains about 0.4 bits per word: k = 9999 at L = 65536 now allows 13.57 instead of 13.11.
- The selection margin dropped from 0.45 to 0.2. Full runs at the new lengths had max round-off
  0.22 or lower.

Single-core full tests after the fix:

| Number | FFT before → after | Time before → after |
|---|---|---|
| 9001·2^662695−1 | 51200 → 49152 | 91.6 s → 85.7 s |
| 8243·2^872226−1 | 71680 → 65536 | 172.6 s → 155.3 s |

## 15:55 — prst against ahnum, alternating with cool-downs

After about 40 minutes of all-core runs the machine slowed: both programs lost 15–25% per test.
Fair comparison therefore needed alternating runs with 3-minute cool-downs. 10 concurrent tests
of 9001·2^662695−1:

| Program | Time per test | Throughput |
|---|---|---|
| prst | 125.2 s, 126.4 s | 283 tests/hour |
| ahnum (table normaliser) | 114.9 s, 121.4 s | 295–313 tests/hour |
| ahnum (compressed weights) | 119.7 s, 123.1 s | 293–298 tests/hour |

Two earlier ahnum runs at about 144 s per test did not recur. In those runs every test was
equally slow, which fits a machine-wide state (heat or background work) rather than ahnum's
memory footprint; the L2 hypothesis was ruled out. The table normaliser stays the default.

Net result: ahnum is about 4–10% ahead of prst on throughput and 5% ahead on a single test
(85.7 s against 90.2 s). prst is x86_64 gwnum translated by Rosetta 2. gwnum's FFTs are hand-written AMD64
assembly, so no native arm64 build is possible and this is the best gwnum available on Apple
Silicon.

## 16:10 — Speed-up plan, items 1 and 2a

**Item 1: why a test slows by about 20–30% when 10 run at once.** Measured on 9001·2^662695−1:

| Condition | ahnum µs/squaring | L1 kernel ns/butterfly |
|---|---|---|
| Alone | 128.1 | 4.95 |
| 9 FP-only spinners (no memory traffic) | 136.2 (+6%) | 5.50 (+11%) |
| 9 other ahnum tests | 153.7 (+20%) | — |

The L1-resident kernel shows the all-core clock drop, about 11%. Contention for shared cache
and memory adds about another 10%. Full 2-minute runs lose more because sustained clocks fall
further with heat.

Shrinking the footprint does not help. With 10 tests, the compressed-weight normaliser (no
1.5 MB table) gives 334 tests/hour against 354 for the table. The likely limit is L2 bandwidth:
each test streams about 3.9 MB per squaring, about 30 GB/s per core and 150 GB/s per 5-core
cluster. Cutting that traffic would cost FP instructions, and the kernels are FP-bound. Little
here is recoverable in code.

**Item 2a: fuse the row-pair transpose into the row FFT.**
- Upper bound first: skipping the transposes entirely saved 7%; skipping the split saved 9%.
- `ahn_dif_r4_rows` and `ahn_dit_r4_rows` are the first and last row-FFT stages (radix 4,
  s = C/4). They read or write the two rows directly and do two adjacent butterflies at once,
  so one 16-byte load covers both columns, and `trn1`/`trn2` happen in registers.
- The row plan now always starts with radix 4, which costs the same per bit as radix 8.

Results:

| FFT length | Before | After |
|---|---|---|
| 28672 | 74.3 µs | 67.4 µs |
| 49152 | 132.7 µs | 120.3 µs |

That is −9.3%, more than the 7% bound for the transposes alone. 9001·2^662695−1 went from
85.7 s to 80.3 s. GMP tests, SME tests and the residue suite all pass.

## 16:30 — Speed-up plan, items 2b, 3, 4 and 5: measured and set aside

- **2b, fusing the real split into the row FFT.** Skipping the split entirely saves 9%, but that
  is its arithmetic: the split kernel has about 34 FP instructions to 9 loads and stores per
  step, so it is FP-bound. Fusion would remove only the memory pass, estimated under 2%, for a
  much more complicated mirrored-butterfly kernel. Not done. The transposes were different
  because they were nearly all memory work.
- **3, finer FFT-length granularity.** Over 4000 random (k < 10000, n of 400K–2M), the chosen L
  averages 2.66% above the continuous minimum safe length (worst 14%). Part of that is the
  tuning table deliberately picking a faster, longer length. This is below the 3% threshold,
  and relaxing the shape rules would touch the normaliser and the transposes. Not done.
- **4, safety margin.** Max round-off over 20,000 squarings at L = 32768 and 98304, k = 3, 255
  and 9999:

  | Margin | Max round-off |
  |---|---|
  | 0.2 (current) | 0.25–0.28 |
  | 0.1 | 0.30–0.31 |
  | 0.0 | 0.31–0.375 |

  Full tests run 30–100× more squarings, which adds a few hundredths. A margin of 0.1 would come
  close to the 0.40 retry threshold for about 0.75% shorter FFTs. Kept at 0.2.
- **5, driver overhead.** The full test of 9001·2^662695−1 averages 121.2 µs per squaring
  against 120.3 µs bare. Gerbicz, the multiplications and setup together cost about 0.75%.
  Nothing worth doing.

Of the five planned items, only the transpose fusion (−9%) paid off. Per core, the NEON
kernels run at the 4-pipe limit and the remaining passes are FP-bound. Under full load, about
half the slowdown is the all-core clock drop and the rest is shared L2 bandwidth.

## 16:35 — Throughput after the transpose fusion

The same routine as before: 10 concurrent tests of 9001·2^662695−1, alternating prst and ahnum,
with a 3-minute cool-down before each run.

| Program | Run 1 | Run 2 |
|---|---|---|
| prst | 117.2 s per test (305.0/h) | 123.5 s (290.3/h) |
| ahnum | 105.0 s per test (339.6/h) | 107.3 s (336.4/h) |

ahnum gives 336–340 tests/hour against prst's 290–305, about 10–16% more throughput. It is
also 11% faster on a single test (80.3 s against 90.2 s). The two ahnum runs agree within 1%;
prst's differ by 5%.

## 17:15 — Second speed-up round: where the cycles go

The last round assumed the kernels run at the 4-pipe limit everywhere. They do in isolation, but
counting the generated instructions (`tools/count_ops.py`) against the measured time says the
whole squaring at 49152 ran at about 75% of that. The distributed compute project was not
running. I misread a load average of ~3 as outside load, but it was my own builds and
benchmarks. From about 17:00 my residue-suite run kept one other core busy, so the later
measurements (the order sweep, the ~4.2 GHz clock estimate, the tangent A/B) shared the cluster
with it. A/B pairs saw the same conditions; the absolute numbers will be repeated with nothing else running.

**Phase profiler.** `AHN_PROF=1` accumulates `cntvct` ticks per phase on thread 0 and prints the
split at exit. Its total agrees with the untimed figure to within ~2%. Measured ceiling under this
load: `frintn`, `fmul`, `fsub` and `trn1` each run at ~16.5 per ns (4 per cycle at ~4.2 GHz).
Against that, at 49152 (R = 96, C = 256):

| Phase | Share of ceiling |
|---|---|
| p1 forward stages | 95% |
| p2 fwd stages, split | 92% |
| p2 inv stages | 87% |
| p1 inverse stages | 80% |
| p2 rows in | 77% |
| normaliser | 80% (see below) |

**Normaliser: latency, not throughput.** With 2 carry chains in flight, each digit's link is
fmla → frintn, about 6.6 cycles, against 16 ops of work, which predicts 2.46 ops/cycle. Outside
error-check squarings it measured 2.45. Rewriting the table record as {osc, 1/base, base·w}
lets each digit be worked in place in one register:

    D = rint(D·osc)·ib;  D += cin·ib;  cin = rint(D);  D = (D − cin)·bw

Every step before the last multiply is a power-of-two scaling of an integer below 2^53, so it is
exact, and (D − cin)·(base·w) rounds the same as the old x·w. The output is bit-identical: maxerr
and residues match on every test number. It is 7 ops instead of 8 (9 instead of 10 with error
tracking), the table is 25% smaller (1.15 MB instead of 1.5 MB at 49152), and with only D and the
carry live per chain, 4 row pairs (8 rows) fit in the 32 registers. In L1 the new kernel runs at
16.1 ops/ns, the full ceiling. 49152 went from 124.6 to 118.7 µs (−4.7%) in back-to-back runs. In
place it still reaches only ~86%; making the table L1-resident (a deliberately wrong build)
recovered just 1 µs, and the last-rows C path costs 0.12 µs, so the remainder is not
explained yet.

Mode b (group 0's first visit) now stores digit/base, which mode c consumes directly, so neither
needs `base` itself.

**Set aside:**
- *Software prefetch of the next pass-1 group:* 2% slower. The hardware prefetcher already
  follows the 4224-byte stride, and the extra prefetches only add L1 pressure.
- *Moving the special row pair (k_r = 0, R/2) out of scalar C:* it costs 0.62 µs (0.5%), against
  ~0.29 µs for a vector pair, so there is ~0.3 µs to gain.

**Radix order matters.** `factorize` puts radix 8 first and the odd radix last, so at 49152 the
column s = 1 stage (the one that loads the group from L2 and applies the four-step twiddle)
was radix 3, which has little compute per byte. Reversing it to [3, 4, 8] saved 3.4 µs (−2.7%).
A sweep of ascending versus default order for the column and row plans:

| L | default | col asc | row asc | both |
|---|---|---|---|---|
| 24576 | 60.8 | 60.8 | 59.8 | 59.4 |
| 40960 | 100.4 | 100.4 | 102.6 | 103.7 |
| 65536 | 170.3 | 163.2 | 169.2 | 163.5 |
| 98304 | 275.5 | 264.5 | 266.4 | 260.6 |
| 131072 | 376.9 | 373.2 | 361.4 | 346.5 |

No order wins everywhere, so the order is now tuned per length. `ahbench tune` times all four
orders interleaved and writes "L us order", the context reads the third column (old two-column
files still parse), `ahn_opts.plan_order` overrides it, and `ahbench --order` exposes that.
`test_gmp` cycles through all four orders, and the residue suite passes in each.

**Twiddle bound for item 2.** Deleting every stage-twiddle multiply (wrong results) gives the upper
bound: 67.6 → 60.0 µs at 28672, 120.3 → 103.7 at 49152, 320.9 → 275.3 at 114688, so 11–14%.
Vector `mov` is eliminated at rename (16 mov+fmla pairs take the time of 16 fmla), so the
tangent form w = c·(1 + iτ) costs 2 FMAs, with c folded into the butterfly's following FMAs.
For DIT stages, where twiddles sit on the inputs, every scale folds: radix 8 drops from 84 to
66 ops, radix 4 from 28 to 22, radix 3 from 22 to 18. τ can be large near cos θ = 0 without hurting
accuracy, because the error stays ~ε·|x|. Only an exact zero needs care (use c = 2^-60). DIF stages
are harder: the trick does not transpose (radix 2 stays at 8 ops). Gaining there means moving
each stage's output twiddles into the next stage's inputs and carrying the leftover scales
down to the four-step table.

## 17:30 — Tangent-form DIT twiddles

New kernels `ahn_dit_r{2,3,4,8}_tan[_b]` and `ahn_dit_r4_rows_tan` take a per-j table of tangents
and scale ratios built by `ahn_tan_table` (src/neon.c). Every DIT stage with s > 1 uses them,
in both pass-1 inverse column stages and pass-2 inverse row stages. Radices 5 and 7 keep the
complex multiply, because folding the scales into their cosine/sine constants needs more scalar
registers than they have free. The DIF (forward) side is unchanged.

Scale bookkeeping in the radix-8 DIT, input scales c_t after x_t (1 + iτ_t):
- Layer 1: (x0, x4) folds c4 → scale 1. (x1, x5), (x2, x6), (x3, x7) fold c5/c1, c6/c2, c7/c3
  → scales c1, c2, c3.
- W8 multiplies lose their √½ multiply. For W8³ the factor is −c·(1 − i), a uniform real
  scale once the sign moves into it.
- Layer 2 folds c2 and c3/c1, or −c3/c1 on the W8³ side. Layer 3 folds c1 or c1·√½.

Every output comes out at unit scale with no extra multiply. Per j that is 7 tangents plus 9
scale scalars, 8 q-registers against the old 7, and the radix-8 constant register is no longer
needed. Op counts per butterfly: radix 8 84 → 66, radix 4 28 → 22, radix 3 22 → 18, radix 2 8 → 6,
fused rows radix 4 56 → 44.

Accuracy: against the scalar reference the new kernels match the old ones (1.8e-15 for radix 8),
including s = 8 and 16, where cos θ = 0 exactly and the table stores c = 2^-60. Max round-off
at the selection boundary over 20,000 squarings is equal or lower everywhere:

| L | k | tangent | old |
|---|---|---|---|
| 32768 | 3 | 0.2500 | 0.2500 |
| 32768 | 255 | 0.2812 | 0.3125 |
| 32768 | 9999 | 0.2500 | 0.2812 |
| 98304 | 3 | 0.2500 | 0.2500 |
| 98304 | 255 | 0.2500 | 0.2539 |
| 98304 | 9999 | 0.2500 | 0.2812 |

Speed in back-to-back runs, with my residue-suite job busy on another core (µs per squaring, `AHN_NO_TAN=1` for the old path):

| Number | FFT | old | tangent |
|---|---|---|---|
| 9357·2^369717−1 | 28672 | 67.3 | 65.5 (−2.7%) |
| 3·2^400000−1 | 24576 | 46.7 | 45.4 (−2.7%) |
| 9001·2^662695−1 | 49152 | 119.5 | 115.9 (−3.0%) |
| 9999·2^1500000−1 | 114688 | 320.5 | 308.5 (−3.7%) |

That is about half of the DIT half of the 11–14% bound, as expected for saving 2 of every 4
twiddle ops.

**Item 4 dropped.** Folding the split's four doublings into FMAs saves 1–2 of its 30 FP ops
(~0.3%), since the factor 2 lands on terms that need a negation to fold. The scalar special pair
is ~0.3 µs. Neither clears the 3% bar.

## 17:50 — Quiet-machine results for this round

Nothing else was running (the distributed compute project was off, and my own residue job was
stopped first). µs per squaring, best of 3, interleaved:

| Number | Original | + normaliser | + tangent DIT | + tuned order |
|---|---|---|---|---|
| 3·2^400000−1 | 47.5 | 45.9 | 44.6 | — |
| 9357·2^369717−1 | 68.6 | 66.4 | 64.6 | 64.3 (now picks 26880) |
| 9001·2^662695−1 | 121.3 | 117.9 | 115.0 | 113.0 (−6.8%) |
| 8243·2^872226−1 | 168.0 | 164.2 | 156.7 | 153.5 (−8.6%) |
| 9999·2^1500000−1 | 324.3 | 315.9 | 304.7 | 298.0 (−8.1%) |

**Re-tune.** `ahbench tune 8192 196608` with all four stage orders took 104 s. The default order is
fastest at only 24 of 169 lengths. Column ascending won 30, row ascending 41, and both ascending
74. The new table is 6–16% faster than the old one length for length (49152: 128.7 → 113.1;
98304: 290.6 → 245.3). It is installed as `~/.ahnum/fft_times.txt` and
`tune/fft_times_m4pro.txt`; the old file is kept in the session scratchpad.

**Full tests.** 9001·2^662695−1 single-core: 74.6 s and 74.9 s, against 80.3 s (−7%). Max round-off
was 0.25, with no Gerbicz errors or retries. 10 concurrent tests, after a 3-minute cool-down:
96.8–97.3 s each, 371 tests/hour, against 336–340 last round (+9–10%). Against the earlier prst
runs (290–305/h) that is ~22–28% more throughput, though prst was not rerun today. The
throughput command logged only the times, so those ten residues were not individually checked.
The single runs matched.

By step: normaliser −2.3 to −3.4%, tangent DIT −2.5 to −4.6%, tuned order −0.5 to −2%. The
loaded-machine order sweep had suggested more at 49152 (−2.7%). I have not measured whether the
tangent kernels changed which order wins there.

Residue suite (28 known residues, n up to 872226) on the final build: orders 2 and 3 pass 28/28.
Order 1 passed 28/28 earlier on the pre-tangent build, and order 0 passes via ctest. test_gmp
cycles through all four orders against GMP.

## 20:37 — Throughput on 9357·2^369717−1 with the new build

`ahbench throughput` squaring-rate samples, 20 s per configuration, quiet machine after a rest:
10 × 1 gives 1314 tests/hour (74.1 µs per squaring per test), 14 × 1 (P + E) gives 1405
(97.1 µs). The first build measured 1125 and 1244. A second round after only a 60 s rest came out
2–6% lower (1283 and 1315), which matches the heat effect seen before. Two earlier attempts this
evening were discarded: one ran right after hours of prst work by the distributed client, and one
overlapped a video call (camera, conferencing and video-encoder processes busy). They gave
inconsistent results, including 14 × 1 slower than 10 × 1.

## 20:47 — Safety margin 0.1: dropped again

With the tangent DIT kernels in, I re-ran the margin sweep at 12 points (L = 32768, 49152, 98304,
196608 × k = 3, 255, 9999), 20,000 squarings each. Max round-off:
- Margin 0.2: 0.25–0.28.
- Margin 0.1: 0.28–0.375, with 0.344 at L = 32768 k = 3 and 0.375 at L = 49152 k = 255.
- Margin 0.0: 0.31–0.375.

Last round's six-point sweep showed 0.30–0.31 at margin 0.1. The wider sweep finds worse points,
and full tests run 30–100× more squarings, so margin 0.1 would sit at the 0.40 retry threshold.
A retry restarts at a larger FFT and costs far more than the ~0.5–1% saving. `BPW_MARGIN` stays
at 0.2. The tangent kernels lowered round-off at margin 0.2 but did not move the safe margin.

## 20:51 — Normaliser fusion: no bound

Fusion would only help if the joins between the inverse stages, the normaliser and the
forward stages cost something: the group's store and reload, or lost overlap between kernels.
I timed pass 1 three ways with a temporary switch (µs per squaring, best of 3, quiet machine):

| Number | all (A) | normaliser skipped (B) | column stages skipped (C) | B + C − A |
|---|---|---|---|---|
| 9001·2^662695−1 | 111.1 | 91.6 | 72.0 | 52.5 |
| 9999·2^1500000−1 | 294.1 | 249.7 | 197.7 | 153.3 |

If the joins are free, B + C − A equals the time outside pass 1. At 49152 that is pass 2, ~53 µs
by the profiler, so the joins cost nothing measurable. The normaliser costs ~19.5 µs whether it
runs after the stages (data in L1) or alone (data from L2). Its ~16% shortfall against the
isolated kernel rate is inside the kernel itself. Dropped. (Skipping the normaliser lets the
values overflow to NaN and inf; Apple's FP units showed no slowdown for that, since B came out
as consistent as A.)

## 20:57 — Forward (DIF) tangent twiddles: bound 5.3–6.4%, shelved

Bound build: every DIF output-twiddle multiply removed (wrong results), DIT tangent kernels kept.
µs per squaring, best of 3, quiet machine:

| Number | full | no forward twiddles |
|---|---|---|
| 9357·2^369717−1 | 62.8 | 58.8 (−6.4%) |
| 9001·2^662695−1 | 110.7 | 103.6 (−6.4%) |
| 8243·2^872226−1 | 150.2 | 141.5 (−5.8%) |
| 9999·2^1500000−1 | 294.4 | 278.7 (−5.3%) |

On the inverse side, the twiddles sat on the butterfly inputs and saved half their ops, which
recovered 45–65% of the bound. On the forward side they have to move to the next stage's inputs:
- That makes p twiddles per butterfly instead of p − 1, so radix 8 saves 12 of 28 twiddle ops
  and radix 4 saves 4 of 12.
- Pass 2's last forward stage leaves a per-element scale that has to be multiplied out before
  the split, about one extra op per point.

Expected net ~1.5–3%, below the 3% bar, for the largest change still on the list. Shelved. The
remaining rows-in item has a ceiling of ~2 µs (~1.7%) at 49152, so no candidate on the list
clears the bar any more.

## 21:07 — New candidates: table traffic under load, and what is left

The throughput workload loses 18% per test at 10 concurrent tests (133.5 µs per squaring against
113 alone). Each squaring streams ~3 MB from L2, about half of it the four-step table (393 KB) and
the normaliser table (1.15 MB). The bound: both tables made L1-resident (every group reuses group
0's slice; wrong results, same instructions). One core: 111.3 → 107.7 µs (~3%). Throughput,
alternating with 90 s cool-downs:

| Run | 10 tests normal | 10 tests hot | 14 tests normal | 14 tests hot |
|---|---|---|---|---|
| 1 | 407.0/h | 409.6/h | 422.1/h | 422.2/h |
| 2 | 397.3/h | 407.5/h | 401.9/h | 416.7/h |

Round 2's normal run fell 2–5% from round 1, which fits the machine warming up, so the bound under
load is 0–2%. Table traffic is not what slows tests down at full load, and rebuilding tables on
the fly would cost FP work. Dropped.

Two more ideas reasoned through and not built:
- **The k penalty.** It comes from the wrapped terms carrying k⁻¹ mod N, and gwnum pays it too.
- **New radices for finer FFT lengths.** The chosen length already averages only 2.66% above the
  minimum.

What is left with real size is off the CPU: an integer NTT on the idle GPU running extra tests
alongside the CPU ones. It is now on the priorities list as a research item; its first step is
a prototype kernel timed at 49152.

## 21:17 — Forward (DIF) tangent twiddles: built, measured, dropped

Design: forward stages stop applying their output twiddles. Each next stage applies them on
its inputs in tangent form, divided by input 0's factor F_0, so it still uses p − 1 tangents per
butterfly. Its outputs carry F_0 as a pending complex factor, folded into the next stage's
per-butterfly table.
- Pass 1: the last stage's pending factors go into a forward copy of the four-step table.
- Pass 2: the pending factor depends only on the column. The last stage (`tana`) gives every
  input its own tangent and leaves a real per-column scale, which the split folds into its e/d
  sums (`ahn_split_sq_s`, +2 ops per iteration).

Kernels: `ahn_dif_r{2,3,4,8}_tanf[_b]`, `_tana`, `_tant4_b`, `ahn_dif_r4_rows_notw`. GMP tests
passed on the first build.

µs per squaring, quiet machine, new path against `AHN_NO_TANF=1`:

| Number | FFT | old | new |
|---|---|---|---|
| 9357·2^369717−1 | 26880 | 63.5 | 63.4 |
| 9001·2^662695−1 | 49152 | 112.4 | 110.7 (−1.5%) |
| 8243·2^872226−1 | 65536 | 151.7 | 152.0 |
| 9999·2^1500000−1 | 114688 | 295.8 | 297.9 (+0.7%) |

Why it fell short of the 1.5–3% estimate (and far short of the 5.3–6.4% bound):
- **The last stages gain ops.** The old last forward stages had no twiddles: radix 8 at 56 ops,
  or 88 with the four-step multiply. They now take over the previous stage's twiddles: 68 for
  `tana`, 98 for `tant4`. The bound build never paid that, and my estimate missed it.
- **Rows in does not care about FP.** Its kernel went from 56 to 32 FP ops, and its time stayed
  at 19.5–19.8 µs (at 114688); it is bound by loading the two rows from L2.
- **The second four-step table costs ~1%.** Before, the forward pass reused the inverse pass's
  table slice while it was still in L1. Pointing the forward pass at the original table (wrong
  results) recovered ~3.2 µs at 114688, but even then pass 1 forward was no faster than the old
  path.

Removing that table would mean also redesigning the inverse so the pending factors fold into the
normaliser weights, worth at most ~1% more. Dropped. The patch is kept in
`docs/experiments/dif-tangent-forward.patch` (applies to 6b24cf7).

## 21:27 — GPU NTT prototype

`gpu/ntt.metal` plus `gpu/ntt_proto.m` (Objective-C host; shaders compiled at run time, since the
offline `metal` compiler is not installed). A batched four-step NTT squaring mod p = 15·2^27 + 1,
N = R × C, in three dispatches that mirror the CPU passes:
1. Column DIF in threadgroup memory, then the four-step twiddle.
2. Row DIF, pointwise square, row DIT.
3. Inverse twiddle and scale, then column DIT.

The arithmetic is Montgomery with 2^32 from 32-bit `mul`/`mulhi`. The kernels are naive radix 2
with a barrier after each stage. Each size checks one squaring against a direct cyclic square at
64 positions. All pass.

| N | numbers in flight | µs per squaring mod one prime |
|---|---|---|
| 32768 | 16 / 64 / 256 | 6.55 / 5.63 / 5.41 |
| 65536 | 32 / 128 | 12.06 / 11.77 |
| 131072 | 64 | 38.84 (a 3.3× jump, not explained yet) |

Cost model for 9001·2^662695−1: the arithmetic is exact, so there is no k penalty and no safety
margin. 45-bit words give 14,727 words, zero-padded to 32768, and the convolution needs ~104 bits,
so 4 primes. That is ~22 µs of NTT per squaring. Combining the primes, carries, reduction mod
k·2^n−1 and splitting back into words are not built.

Alongside the CPU (14 tests of 9001·2^662695−1, 90 s cool-downs between runs):

| | CPU tests/hour | GPU µs per prime-squaring |
|---|---|---|
| alone | 452.2 | 5.57 (99% busy) |
| together | 352.7 (−22%) | 6.21 (92% busy) |

At 6.21 µs, and guessing 10–20 µs for the unbuilt parts, the GPU would add ~120–150 tests/hour, so
together ≈ 490 against 452 alone, about +8% (+2% to +26% depending on those parts). The CPU loses
~100 tests/hour when the GPU runs, probably from the shared power budget, possibly also memory
bandwidth. The GPU's feeder thread competes with 14 CPU tests for 14 cores. Next: separate the
power and bandwidth causes, try 13 CPU tests plus the feeder, and optimise the NTT kernel to
see if a cheaper GPU workload costs the CPU less.

## 21:47 — GPU alongside the CPU: the cost is power, not the feeder thread

I reran with the shader and binaries frozen in a separate directory. The first attempt read
`gpu/ntt.metal` at start-up while I was editing it, so its GPU half never compiled and its
results were discarded. 9001·2^662695−1, `ahbench throughput` 30 s, 90 s cool-downs, GPU at
N = 32768 with 256 numbers in flight:

| Run | CPU tests/hour | GPU µs per prime-squaring (busy) |
|---|---|---|
| 14 tests alone | 441.7 | — |
| 14 tests + GPU at 50% | 409.2 (−7.4%) | 14.56 wall (46%) |
| 13 tests alone | 422.6 | — |
| 13 tests + GPU full | 330.4 (−21.8%) | 5.86 (97%) |
| 14 tests + GPU full | 329.6 (−25.4%) | 6.19 (93%) |

With 13 CPU tests the GPU runs at almost its solo speed (5.86 against 5.57), so with 14 the
feeder thread was being starved. The CPU loss stays at ~22%, though, and scales with how busy
the GPU is: −7.4% at half load, −22 to −25% at full. That fits the shared power budget.

Combined, using the cost model (4 primes at N = 32768 plus a guessed ~15 µs per squaring for the
unbuilt prime-combining, carries and reduction):
- 13 tests + full GPU: 330 + ~141 ≈ 471 tests/hour.
- 14 tests + half GPU: 409 + ~62 ≈ 471.
- 14 tests + full GPU: 330 + ~137 ≈ 467.

All about +6–7% over the best CPU-only figure (442). The CPU's loss tracks the GPU's busy time,
not its work done, so a faster GPU kernel gives more GPU tests for the same CPU cost. GPU kernel
efficiency is therefore the lever. If the GPU did the same work in half the time, the same run
would come out near +25%.

**Fused radix-2² stages: no help.** Fusing pairs of radix-2 stages into one step (one barrier
and one threadgroup round trip per pair, `AHN_GPU_FUSE=1`) is correct but slower at C = 256
(32768: 5.69 → 6.60 µs; 256 × 256: 11.86 → 15.50 µs). It only helps at C = 512 (18.2 → 13.4),
where the plain kernel was poorly placed. Fusion keeps every modular multiply, so this points
to the integer multiplies as the limit: ~2.4M `mul`/`mulhi` per prime-squaring. Next to try:
Shoup's precomputed-quotient multiply for twiddles (3 multiplies instead of Montgomery's 4).

## 21:50 — GPU kernel: two more hypotheses ruled out

The fusion result suggested the integer multiplies were the limit. Shoup's precomputed-quotient
multiply for all twiddles (tables of (w, ⌊w·2^32/p⌋) pairs, 3 multiplies instead of Montgomery's
4) made it **twice as slow**: 5.53 → 10.32 µs at 32768 and 11.76 → 22.06 at 65536. Combined with
fusion it came back to 5.88. So the multiplies are not the limit, and the fusion reasoning was
wrong. Doubling the twiddle bytes doubled the time, which looked like twiddle loads from device
memory. Copying each stage's twiddles into threadgroup memory at kernel start changed little
(5.53 → 5.31 at 32768, 11.76 → 12.32 at 65536), and Shoup stayed 2× slower with on-chip
twiddles. Both changes are reverted (the kernel stays as committed in e9db715).

What limits the kernel is not known. Timings alone have not found it: three hypotheses (barriers,
multiplies, twiddle loads) are ruled out. GPU performance counters (Xcode's Metal System Trace)
would show it directly, but Xcode is not installed on this machine.

## 22:06 — GPU counters: the NTT kernel is latency-bound

Xcode is now installed. The default Metal System Trace records only one GPU counter (RT unit),
and the counter set cannot be chosen from the command line. A template saved from Instruments
with the Metal GPU Counters track set to "Performance Limiters" (`~/metal-limiters.tracetemplate`)
records 80 counters with `xctrace record --template`. Averaged over the 9017 samples where
compute kernels ran (prototype at N = 32768, 256 numbers in flight):

| Counter | Value |
|---|---|
| Instruction throughput limiter / utilization | 75% / 21% |
| Integer and complex limiter / utilization | 54% / 39% |
| Integer and conditional limiter / utilization | 51% / 43% |
| ALU utilization | 31% |
| Kernel occupancy | 88% |
| Compute shader launch limiter / utilization | 96% / 2% |
| Threadgroup memory read / write bandwidth | 709 / 709 GiB/s |
| GPU (DRAM) bandwidth | 53 GB/s |

No execution unit is saturated and DRAM is far from its limit. Instructions are queued but
mostly stalled, and the cores are full, so new threadgroups cannot launch. Each thread does one
butterfly (a few integer ops) between threadgroup-memory round trips and barriers, 8–15 times per
transform, and 88% occupancy is not enough to hide that latency. This explains the earlier
results:
- Shoup multiplies cut ALU work, which was not the limit, and added memory traffic.
- Fused radix-2² steps still did little work per barrier.

The usual GPU FFT layout should fix it: each thread holds 8–16 elements in registers, runs a
radix-8/16 sub-transform there, and exchanges through threadgroup memory 2–3 times per
transform instead of 8–15.

## 22:09 — Register-based GPU kernels: −12%, now limited by register spills

`gpu/ntt_reg.metal` (`AHN_GPU_REG=1`): each length L splits as A × B. A thread holds B elements
in registers and runs a length-B DFT there (unrolled radix-2 network). After the inner twiddle
w_L^(a·kb) and one exchange through threadgroup memory, it runs the length-A DFTs on the other
index. Two barriers per row square instead of one per radix-2 stage. Rows are read and written in
natural order. The column pass stores rows in natural frequency order, so the four-step table is
w_N^(k1·n2). Correct on the first run (same direct-square check).

| N | radix-2 kernels | register kernels |
|---|---|---|
| 32768 (128 × 256: 8×16 columns, 16×16 rows) | 5.55 µs | 4.87 µs (−12%) |
| 65536 (256 × 256) | 11.78 µs | 10.20 µs (−13%) |

Counters for the register kernels against radix-2:

| Counter | radix-2 | register |
|---|---|---|
| Threadgroup memory read / write | 709 / 709 GiB/s | 94 / 94 GiB/s |
| Kernel occupancy | 88% | 31% |
| Occupancy manager target | 100% | 47% |
| L1 eviction rate | 25% | 61% |
| Register traffic through L1 | ~0 | 72 GiB/s |
| Integer and complex limiter / utilization | 54% / 39% | 86% / 45% |

The exchange traffic went as planned, but 16 elements per thread, plus the radix-16 network's
temporaries, overflow the register file. Values spill to L1, and the occupancy manager cuts
occupancy to a third. Forcing full unrolling on every register-array loop changed nothing
(4.88 µs), so the arrays were already in registers; the pressure is genuine. Next: 8 elements per
thread with a three-step split (C = 256 as 8 × 8 × 4), one more exchange, but small enough to stay
in registers at full occupancy.

## 22:13 — GPU: Stockham radix-8 loses, Shoup in the register kernels wins (−21%)

**Stockham, 8 elements per thread.** Radix-8 passes (the last one radix 4 or 2), natural order in
and out, two ping-pong threadgroup buffers, one barrier per pass. Correct, but slower than the
16-per-thread register kernels: 7.56 µs at 32768 at first. The per-pass index arithmetic divided by
a runtime sub-length, and the GPU has no hardware integer divide; with shifts and masks instead it
reached 6.86, still behind 4.87. Counters: occupancy 40% with the occupancy target at 100% and low
evictions. That cap comes from threadgroup memory (two buffers per threadgroup), not registers.
The integer "complex" pipe showed an 82% limiter against 47% utilization, the same pattern as the
register kernels (86% / 45%). Removed.

**Shoup in the register kernels.** The complex pipe handles the costly integer ops, probably
including `mulhi`. Montgomery uses two `mulhi` per multiply, Shoup one (plus a second table word).
In the radix-2 kernel Shoup was 2× slower, because that kernel was bound by threadgroup round
trips and the bigger table cost more than the saved `mulhi`. In the register kernels the complex
pipe is the limit:

| N | radix-2 | register | register + Shoup |
|---|---|---|---|
| 32768 | 5.55 µs | 4.87 µs | 3.83 µs |
| 65536 | 11.78 µs | 10.18 µs | 7.93 µs |

So −31% against where the GPU work started. The same optimisation went from 2× slower to 21%
faster once the limit moved.

Cost model update for 9001·2^662695−1 (4 primes at 32768, plus a guessed ~15 µs per squaring for
the unbuilt parts): ~30 µs per squaring on an idle GPU. Alongside 13 CPU tests at the measured
−22% CPU cost (if the GPU's slowdown under CPU load stays ~10%), that is ~330 CPU + ~170 GPU ≈
500 tests/hour, about +13% over the CPU alone (442), up from ~+7%.

## 22:20 — Full GPU squaring mod k·2^n−1: correct, 30.6 µs per squaring

`gpu/gpu_sq.m` plus `gpu/sq.metal` (appended to `ntt_reg.metal`). A complete squaring step on
the GPU for B numbers at once, with no CPU work between squarings:
1. `to_res`: 45-bit words → residues mod four NTT primes (15·2^27+1, 27·2^26+1, 7·2^26+1,
   45·2^24+1), zero-padded to 32768.
2. Four NTT squarings: register kernels with Shoup twiddles, three dispatches each.
3. `crt_split`: Garner CRT to a 128-bit coefficient (< 2^104), split into three 45-bit parts.
4. `carry`: product words.
5. `redc` and `fold`.

**The reduction mod N = k·2^n−1.** Numbers are kept in Montgomery form with R = 2^n. Since
N ≡ −1 (mod 2^n), Montgomery reduction of a product P = A + B·2^n (A < 2^n) is just
P·2^−n ≡ **B + k·A** (mod N): a word-parallel multiply by the small k plus an add, no division.
The result can reach 2^(n+29) and the input to the next square must stay bounded, so a
one-thread fold takes T = bits above n, writes T = q·k + r, and replaces T·2^n by
q + r·2^n (since k·2^n ≡ 1). That keeps values under 2^(n+14). The squaring step is therefore
x → x²·2^−n ≡ x²·k (mod N), which the check reproduces on the CPU with ahnum
(`ahn_square`, then `ahn_mul_small(k)`).

**Results.** Correct on the first run at 3·2^20000−1, and at 9001·2^662695−1 after 20 squarings
(14,728 words, NP = 32768, no carry overflows). Throughput: 30.6 µs per squaring with 32 or 128
numbers in flight; sustained 31.4 µs, so ~20 s of GPU time per test and 173 tests/hour from
the GPU alone. The earlier estimate (4 × 3.83 µs NTT + ~15 µs other) was close: the CRT, carries
and reduction cost about as much as the four NTTs.

Not handled yet: a carry that ripples past one word (counted, never seen in testing), Gerbicz
checks, and checkpoints.

## 22:26 — CPU + GPU together: +22% throughput

9001·2^662695−1, frozen binaries, `ahbench throughput` 30 s, full GPU squaring (`gpu_sq`, 64 in
flight) running alongside, 90 s cool-downs:

| Run | CPU tests/hour | GPU tests/hour | Total |
|---|---|---|---|
| 14 CPU tests alone | 455.7 | — | 455.7 |
| 13 CPU tests + GPU | 382.1 | 174 (31.3 µs, 99% busy) | 556 (+22%) |
| 14 CPU tests + GPU | 374.2 | 158 (34.3 µs, 92% busy) | 532 (+17%) |

Leaving one core for the GPU's feeder thread is better: with 14 CPU tests the feeder is
starved and the GPU drops to 92% busy. This went better than the NTT-only prototype predicted,
which cut the CPU by ~22% at full GPU load. With the full squaring kernel the CPU loses ~10%
(382 against ~423 for 13 tests alone, measured earlier this evening), and the GPU loses nothing.
The full squaring spends half its time in the CRT, carry and reduction kernels, and those
presumably draw less power than the NTT. This is a measured throughput rate, not full tests:
the GPU path still lacks Gerbicz checks, checkpoints and handling for a carry rippling past one
word.

## 22:40 — GPU PRP driver: carry fix-up, Gerbicz, checkpoints, known residues match

**Carry ripple.** The one-word carry leaves a word over 45 bits only when a carry ripples
further, which never happened in testing but is possible. That would be a deterministic error:
Gerbicz would catch it, but rolling back would reproduce it every time. Now any overflow sets a
per-number flag, and a one-thread-per-number `fixup_carry` pass normalises that number
sequentially (it costs one flag read when nothing overflowed). It runs after the product
words and after the reduction. A test mode (`AHN_GPU_FORCE_RIPPLE=1`) skips the fast one-word
carry and flags every number, so the sequential path does all the carrying. It matches the CPU
at 3·2^20000−1 and at 9001·2^662695−1. Normal speed is unchanged (31.5 µs per squaring).

**Shared host code.** `gpu/gpu_ctx.h` holds the setup (pipelines, per-prime tables, buffers),
`gctx_sqr` and `gctx_mul` (a general Montgomery product: both operands through the column pass,
a new two-operand row kernel `row_mul_r`, then the usual CRT, carries and reduction), and the
host-side conversions. `gpu_sq.m` (benchmark) and `gpu_prp.m` (driver) use it.

**Driver** (`gpu/gpu_prp.m`). It mirrors `ahn_prp`:
- x_0 = 3^k is computed on the CPU and converted to Montgomery form (x·2^n ≡ q + r·2^n for
  x = q·k + r).
- Gerbicz block L = √n with checks every L/4 blocks; on failure it rolls back to the last
  verified (i, x, d), and the < L tail is run twice and compared.
- The result is converted back from Montgomery form (x = X·k, since 2^−n ≡ k) and finished
  with the CPU's `ahn_residue_nm1`, now exported from prp.c.

There is no round-off check, because the integer NTT is exact; Gerbicz covers hardware faults
and bugs. Checkpoints are written atomically and resume automatically. All B slots run the same
number: the point is to measure throughput and check end to end.

| Test | Result |
|---|---|
| 1203·2^90001−1, 4 slots | RES64 4BE9F1953E350085, matches, 6.9 s |
| same, fault injected at 50,100 | caught at the next check (67,500), rolled back to 45,000, matches |
| same, killed mid-run then resumed | resumed at 45,000, matches |
| 9357·2^369717−1, 16 slots | F5FDA01218F20439, matches, 11.0 s per test, 327 tests/hour |

At n = 369717 the 45-bit word count (8217) lands just past half of 16384, so the padded length
doubles to 32768. 46-bit words would fit 16384 (8038 words, a coefficient bound of 2^105 against
the primes' 2^119.9). The word size should be chosen per n.

## 23:20 — 14-test full-run throughput for the speed history

The README speed history tracked only 10-test full runs (371/h at `3631fc5`). The 456/h figure
from earlier tonight is a 30 s `ahbench throughput` squaring-rate sample with 14 tests. It is not
comparable: rate samples read ~10% above complete runs. So I measured the same way as the 10-test
column, after a 3-minute cool-down: 14 concurrent `ahbench prp 9001 662695` runs, each checked
against the expected residue. All 14 were probable prime with RES64 0000000000000001, 0 Gerbicz
errors and max round-off 0.25, at 123.2–124.0 s per test (mean 123.6 s), so
**14 × 3600 / 123.6 = 408 tests/hour** (+10% over 10 tests). The load log peaked at 18.7, a few
threads above the 14 tests (`duetexpertd` held a core), worth perhaps 2–4%.

The README table gained a 14-test column. The first build (321/h, 156 s per test) and prst
(329.4/h) come from the 13:35 and 14:40 entries; the intermediate versions were not measured with
14 tests.

Two attempts before that never started. In zsh a glob that matches nothing is an error, so
`rm -f dir/f14_*.log` aborted the command before the cool-down. Then a backgrounded
`cd && S=... && sleep && (logger) &` chain took the `cd` and the variable into its subshell,
away from the foreground part that launched the tests. A script file fixed both.

Earlier in the evening the CPU + GPU end-to-end run hit heavy outside contention: a 15-minute
load average of ~32 against ~14 threads of our own. All 64 GPU tests of 9001·2^662695−1 were
correct (probable prime, RES64 1, Gerbicz errors 0) at 23.3 s per test, 154 tests/hour, but the
CPU side read 91 tests/hour, so the combined figure needs a quiet rerun.

## 2026-10-12 08:15 — Guarding benchmarks against outside processes

The 408 tests/hour run lost perhaps 2–4% to `duetexpertd`, the macOS daemon behind Siri
Suggestions and proactive suggestions. It rebuilds its prediction models in bursts when the
machine looks idle, and a box running only command-line tests looks idle to it. It runs as a
per-user LaunchAgent and had used 65 minutes of CPU time since it last started. Rather than
chase that one daemon (Spotlight indexing, photo analysis, Time Machine and software update do
the same), `tools/quiet.sh` checks for any outside process:

- `check [SETTLE] [TIMEOUT]` waits until nothing outside has been busy for SETTLE seconds
  (default 180, which doubles as the cool-down), or exits 1 listing the busy processes.
- `log FILE` runs in the background during a measurement and records the load average and every
  outside process at 5%+ CPU every 5 s; `summary FILE` gives the peak load and each process's peak.

The first thresholds (5% per process, 25% in total) never passed with the display on:
WindowServer sits at 6–17% and a Chrome renderer at ~10%. A process at 10% takes a tenth of a
core, about 0.7% of 14-test throughput; `duetexpertd` took a whole core. So `check` now waits on
any process above 20% or a total above 50%, and `log` still records everything from 5%, so the
journal can say what ran alongside each measurement.

## 2026-10-12 09:40 — Quiet-machine round: 14 CPU, CPU + GPU, GPU alone

Distributed compute stopped, `tools/quiet.sh check` before each phase (3 quiet minutes), the
load logged throughout, 9001·2^662695−1 at `999a580`, every residue checked.

- **14 CPU tests:** all correct, 118.7–119.4 s (mean 119.1 s), **423 tests/hour**. The 408/h run
  with `duetexpertd` holding a core was 3.7% low. Outside processes peaked at 27% of a core.
- **GPU alone, 64 slots:** all correct, 0 Gerbicz errors, 1330.5 s, **173 tests/hour**, the same
  as the squaring-rate sample. `gpu_prp` itself used ~0% CPU (it waits in `waitUntilCompleted`)
  and the load average stayed under 1.6.
- **13 CPU + GPU:** the GPU batch was correct at 1407 s (164 tests/hour, 5% below alone). All 39
  completed CPU tests were correct, but they slowed down from round to round: mean 159.5 s, then
  249 s, then 522 s. The load average rose from 13 to 54.7 while outside processes logged only
  7–17% of a core, `ahbench` runs one thread and `gpu_prp` one, memory was 87% free and no
  thermal warning was recorded. The display turned off at 08:54, about when round 3 began and
  the load passed 40. Last night's combined run (load ~32, CPU side 91 tests/hour) looks like the
  same effect rather than outside load. Round 1 alone gives 13 × 3600 / 159.5 + 164 = 457
  tests/hour (+8% over 423), well below the +22% from 30 s samples, and even that holds only if
  the slowdown can be fixed.

Two things went wrong with the method. The Mac's system sleep timer is 1 minute, and with the
compute client stopped nothing held it awake: it slept from 08:33 to 08:41 during a pre-flight
wait. Benchmark processes do not prevent idle sleep, so runs now need `caffeinate -i`. And the
logger only recorded outside processes, so it could not show how much CPU our own processes got
while the load climbed; it now logs their total as well.

## 2026-10-12 10:35 — CPU + GPU rerun with the display kept on

Same run as 09:40's middle phase under `caffeinate -i -d`, plus a 30 s sampler of runnable
threads per process and each `ahbench`'s %CPU. The display was not the cause: rounds averaged
157 s, 221 s and 494 s again (GPU 1415.6 s, 163 tests/hour, all 64 + 39 residues correct).

The sampler shows what happened. For the first ~4 minutes the 13 `ahbench` processes got 1226–
1298% CPU (each ≥92%). From 10:14 the total fell to 900%, then held at 520–630% for the rest
of the run, with single tests down to 32–45%. Outside processes used 5–15% and `gpu_prp` ~0%.
So about 6 of the 14 cores were doing our work and nothing else was using the other 8, while
the load average (runnable threads) sat at ~20: the tests were runnable but not being run.
No thermal or performance warning was recorded by `pmset -g therm`.

The machine is a 16" MacBook Pro (Mac16,7, M4 Pro, 10 P + 4 E cores) on AC. The slowdown starts
a few minutes into sustained load and does not recover, which looks like thermal or power
management taking cores away. Every CPU-only measurement in the speed history lasts ~2 minutes
per test with all tests in step, short enough to finish before it starts. Next: a CPU-only run
of the same length to see whether the GPU is needed to trigger it, with `powermetrics` reading
frequency, residency and thermal pressure directly.

## 2026-10-12 11:30 — 25 minutes of CPU-only load, with powermetrics

14 `ahbench prp 9001 662695` slots kept busy for 1500 s, no GPU, `sudo powermetrics -i 10000
--samplers cpu_power,gpu_power,thermal` alongside. All 154 completed tests were correct.

No cores were lost: the 14 tests held 1300–1376% CPU and the load stayed at ~15.5. But they got
slower each round: 116.7, 124.0, 126.8, 128.6, 130.4, 130.6, 131.5, 131.7, 132.6, 136.3,
135.4 s. powermetrics shows why. Both P-clusters ran at 2948–3016 MHz with ~38.5 W of CPU power
in the first minutes, then fell steadily to ~2650 MHz and ~31 W (E-cluster fixed at 2592 MHz).
Thermal pressure went Nominal → Moderate after ~12 minutes → Heavy after ~17. The 10% clock
drop matches the 13–16% slowdown. `pmset -g therm` reported none of this; the pressure level is
readable without root as `notifyutil -g com.apple.system.thermalpressurelevel` (0 nominal,
1 moderate, 2 heavy).

So the README's CPU numbers are 2-minute figures from a cool machine. Sustained 14-test
throughput on this MacBook Pro is 14 × 3600 / ~135.5 = ~372 tests/hour, not 423. And the
collapse seen in both CPU + GPU runs (whole cores idle while tests wait) did not happen
without the GPU. The likely explanation, not yet measured, is that GPU power on top of ~31 W of
CPU pushes thermal pressure past Heavy, where macOS stops scheduling work on some cores rather
than only lowering clocks. If so, running both on this laptop is a net loss after ~6 minutes.

`tools/quiet.sh check` now waits for thermal pressure 0 and `log` records the level, since a
warm start lowers the clocks by up to 10%.

## 2026-10-12 12:05 — CPU + GPU with powermetrics: macOS gives the power to the GPU

Same 13 CPU + 64-slot GPU run, with `powermetrics` (10 s samples). All residues correct
(64 GPU, 39 CPU). GPU 1446.3 s, 159 tests/hour. CPU rounds: 181 s, 506 s, 588 s.

| time | P-cluster MHz | CPU W | GPU W | pressure | `ahbench` total CPU |
|---|---|---|---|---|---|
| 11:36:20 (start) | 3026 / 3138 | 40.6 | 21.9 | Nominal | — |
| 11:37:20 | 1692 / 1812 | 14.8 | 22.0 | Nominal | 1193% |
| 11:39:21 | 1800 / 1800 | 11.9 | 19.3 | Heavy | 719% |
| 11:41–12:00 | ~1798 / ~1796 | 4.1–4.9 | 14–17 | mostly Heavy | 490–570% |

Within a minute of the GPU starting, and while thermal pressure was still Nominal, the P-cluster
clocks were capped at 1800 MHz (from ~3000) and CPU power cut from 40.6 W to 14.8 W. The GPU
kept 22 W. Once pressure reached Heavy, the cap stayed and the P-cores were also kept idle much
of the time: in a sample at 11:50 the four E-cores were 100% active at 2592 MHz, but the ten
P-cores were active 81, 42, 27, 35, 17, 80, 36, 26, 25 and 11% (3.8 cores' worth) on 4.3 W
while the GPU drew 15.5 W. The runnable `ahbench` threads waited for P-cores that were held
idle, which is the load of ~20 with little CPU in use.

So this is macOS's power policy on this MacBook Pro, not outside load. With the GPU busy the
CPU is limited to a few watts, and the whole package runs at ~20 W, well below the ~31 W the
CPU alone sustains. Sustained throughput: CPU side 13 × 3600 / ~550 s ≈ 85 tests/hour + GPU 159
≈ **244 tests/hour**, against ~372 for the CPU alone and 173 for the GPU alone. Running both is
a 35% loss here. The +22% from 30 s samples measured the first minute, before the cap.

## 2026-10-12 13:25 — Decision: CPU only for now

The GPU path goes on hold. On a laptop macOS gives the power budget to the GPU and the CPU
loses more than the GPU adds, so it would only pay on desktops, and that makes it lower
priority than CPU work that helps every machine. It may be revisited on the Mac mini. The
README now gives complete-test figures for the GPU rows, labels the speed history as
one-round (~2 minute) figures, and adds the sustained 14-test number (~372 tests/hour); the
priorities file moves the GPU item to an "On hold" section with these measurements.

## 2026-10-12 19:40 — Next CPU candidates

With the GPU on hold, the candidate list had one item left, below the bar (pass-2 rows in,
~1.7%). Today's runs change what to rank against: the workload is 14 tests sustained on a
power-limited laptop (P-cores ~2650 MHz, ~31 W), but every phase share and FP-limit figure in
the file comes from one quiet core at ~4.2 GHz. The new list in
`speed-improvement-priorities.md` starts with a profile under sustained load (step 0), then:
test count and placement under the power cap (measurement only), per-test cache footprint for
the 4 MB E-cluster L2, the pass-1 inverse stages (80% of the FP limit against 95% forward), the
normaliser's carry-chain latency, and radix-16 stages to cut FP work in pass 1. None has a
measured upper bound yet; each gets one before any code is written.

## 2026-10-12 21:00 — Profile under load, and 14 vs 12 tests sustained

Quiet single test, `AHN_PROF=1`: 75.5 s, 112.95 µs per squaring, the same split as before.
Then 14 and 12 tests kept busy for 25 minutes each (quiet check and thermal pressure 0 before
each), slot 1 profiled on every test. All 154 + 144 completed tests correct.

Profile of slot 1, µs per squaring:

| Phase | Quiet | 14 tests, round 1 | round 6 | round 11 | round 11 / quiet |
|---|---|---|---|---|---|
| p1 inverse stages | 20.44 | 33.99 | 37.48 | 39.15 | 1.92 |
| p1 normalise | 17.02 | 28.94 | 32.08 | 33.41 | 1.96 |
| p1 forward stages | 19.13 | 29.83 | 32.87 | 34.22 | 1.79 |
| p2 rows in | 8.39 | 13.80 | 15.23 | 15.92 | 1.90 |
| p2 fwd stages | 13.63 | 21.60 | 23.97 | 25.01 | 1.83 |
| p2 split | 13.01 | 19.53 | 21.64 | 22.54 | 1.73 |
| p2 inv stages | 12.15 | 19.96 | 22.21 | 23.13 | 1.90 |
| p2 rows out | 7.35 | 12.11 | 13.36 | 13.93 | 1.90 |
| total | 112.95 | 182.79 | 202.20 | 210.80 | 1.87 |

Every phase slows by about the same factor. The FP-bound ones slow least, as they would with a
lower clock; the normaliser (latency-bound), the p1 inverse stages and the pass-2 row loads
slow most, so they gain a little share. No phase becomes a new bottleneck, and memory traffic
costs only a little more under load, which fits the 0–2% table bound from 2026-10-11.

Test count: 14 tests settle at 366–372 tests/hour (137 s each by round 11), 12 tests at 344–347
(124.5 s each). The faster tests do not make up for two fewer, so the E-cores are worth running
even under the power cap. The 12-test run started 4 minutes after the 14-test one, warmer, but
the settled rounds compare like with like. The 10-test run was dropped: the trend was clear,
and the scheduler moves tests between clusters, so 10 tests would not have meant P-cores only.

The priorities file now carries the under-load profile, drops items 0 and 1, and keeps the
cache footprint, p1 inverse stages, normaliser and radix-16 items in that order.
On reflection the cache-footprint item goes to the bottom of the list: the under-load profile
is the evidence against it. Next is the pass-1 inverse stages, the largest phase (18.6% under
load) and at ~80% of the FP limit against 95% for the forward stages of the same pass.

## 2026-10-12 21:03 — Pass-1 inverse stages: the gap is the group load

The inverse stages are the first to touch each pass-1 group after pass 2, so their first stage
loads the group's 12 KB (96 rows × 128 B, 4224-byte stride) from L2; the forward stages run on
data the normaliser has just left in L1. A temporary switch read each group into L1 before
its inverse stages (results stay correct; the read was charged to the "phase b" bucket).
One core, `ahbench time 9001 662695 --iters 20000`, three alternating pairs, quiet machine:

| | p1 inverse | p1 forward | read | total |
|---|---|---|---|---|
| normal | 20.42–20.53 µs | 18.93–19.03 | — | 114.0–114.7 |
| read first | 18.41–18.45 | 18.96–19.00 | 2.87–2.89 | 114.9–115.3 |

On hot data the inverse stages are as fast as the forward ones, so there is nothing to fix in
their code. The cold load costs them 2.1 µs; doing it separately costs 2.9 µs, so the stages
already hide part of it. The ceiling is 2.1 µs (1.8%) even if the load were free, below the 3%
bar. Dropped; the patch is in `docs/experiments/p1-touch-before-inverse.patch`. The "80% of the
FP limit" figure was this load, not slow kernels. The normaliser is next.

## 2026-10-12 21:06 — Normaliser: at the FP limit on hot data

The priorities file said the normaliser was still tied to its carry-chain latency. Counting
`ahn_norm_ao`'s loop (`tools/count_ops.py`): 576 vector ops per 128 digits (448 FP + 128
`trn`), 704 with the round-off check. A microbenchmark (`docs/experiments/normbench.c`) runs the
kernel on one group with its data and table in L1:

| Kernel | ns per 128 digits | ops/ns | µs per squaring |
|---|---|---|---|
| `ao` | 34.0–34.3 | 16.9 | 13.1 |
| `aoe` | 40.8–42.1 | 17.3 | 15.7 |

Both run at the measured FP ceiling (~16.5 ops/ns), so eight chains in flight already hide the
fmla → frintn latency. With one squaring in four checking round-off the floor is ~13.7 µs; in
place it takes 17.0–18.4. The gap is the 1.15 MB normaliser table streamed from L2 (the group's
data is still in L1 from the inverse stages), which the 2026-10-11 L1-resident-tables run
already bounded at 0–2% under load. Dropped. The `trn` shuffles are 22% of its ops, but they
follow from the layout: the pass-1 FFT keeps one complex value per register, and the
normaliser needs two rows' digits per register for two chains.

That leaves one candidate on the list with a ceiling near 3%: radix-16 stages to cut FP work.
Two of the three items ranked above it today turned out to be memory loads already measured,
not slow code.

## 2026-10-12 21:12 — Radix-16: counted, no saving

Op counts of the generated kernels (`tools/count_ops.py`), vector ops per butterfly: radix-8
no-twiddle 56, tangent 66; radix-4 no-twiddle 16, tangent 22. One complex value per register,
so the counts track real flops. A 16-point DFT needs 168 real flops (144 adds, 24 multiplies),
2.63 per point per bit, against 2.33 for 8 points. Radix-16 cuts the outer twiddles from 0.29
to 0.23 per point per bit, about 0.08 ops at ~1.4 per tangent twiddle, which does not cover the
butterfly's extra 13%. For the 256-point pass-2 rows, today's 8 × 8 × 4 costs ≈ 20.75 ops per
point and 16 × 16 ≈ 21.1. The sweep it would remove runs over rows in L1. Dropped.

The earlier note in the priorities file, that 16 complex values would not fit NEON's registers,
was wrong: the kernels hold one complex value per q register, so 16 fit with room to spare. It
was not the reason to drop it.

The list is now down to items with measured or estimated ceilings of 0–2% each: per-test
cache footprint (tables), pass-2 rows in (~1.7%), and the pass-1 group load (1.8%, but a
separate read costs more than it saves). The kernels run at 80–95% of the FP limit and the
rest is L2 traffic that is cheap under load. The next gains need either a lower bar with
several 1–2% items stacked, or a change at the algorithm level.

## 2026-10-12 21:25 — Literature check: the k penalty, proofs, error bounds

With the kernel list empty, a search for techniques at the algorithm level.

**The k penalty is inherent to the IBDWT for k·2^n±1.** Percival (Math. Comp. 72 (2003),
387–395, "Rapid multiplication modulo the sum and difference of highly composite numbers") is
the source of the method ahnum uses. Its worst-case bound for k·2^m ± 1 has
|ax| < sqrt((k·2^m)^(1/N) · 3(k² − 1)/(2 log k² log 4)): the weighted digits grow with k, so
the error grows with k and the usable word size shrinks by about ½·log2 k bits, which is what
ahnum's fitted 0.525·log2 k shows. The paper's own example: an FFT of 2^18 suffices up to m = 6M
for k = 3 but only 3M for k = 557. Nothing found removes it within the IBDWT. The known
alternative avoids the k-weighting by computing the full product with a Mersenne-style
transform (no k term, ~20.4 bits per word here) and reducing with x·2^n ≡ x/k, i.e.
x² = H·2^n + L ⇒ x²·2^−n ≡ H + k·L. That needs about 2(n + log2 k) bits of transform against
n bits at 20.4 − 0.525·log2 k, so it wins only above log2 k ≈ 19–20. That matches what LLR-family
software does: an alternative LLR implementation on mersenneforum uses zero-padded FFTs for k of
22–45 bits and generic reduction beyond. At k = 9001 (13 bits) the IBDWT stays the faster choice.

**Proofs instead of double-checks.** Li and Gallot (arXiv:2209.15623, "An Efficient Modular
Exponentiation Proof Scheme") generalise the Gerbicz–Pietrzak proof from power-of-two exponents
to any modular exponentiation, so it covers a Fermat PRP of k·2^n − 1 (exponent N − 1). They
report negligible prover overhead and that at PrimeGrid it doubled the efficiency of the
distributed searches, by replacing full double-checks with certificate checks. PRST implements
proof generation and certificate verification; LLR2 is archived in its favour. This does not
speed up a squaring. For a search that double-checks its results, a proof-producing ahnum would
roughly double useful output per machine; for one that already uses PRST's certificates, ahnum
needs to produce certificates PRST accepts before it can take that work.

**Error bounds.** Percival's proven bounds are worst-case; his measurements put random-input
errors 1.6–8.7 bits below the "unlucky" inputs as N grows from 2^4 to 2^18. GIMPS and gwnum, like
ahnum, choose word sizes from measured errors on real inputs and rely on Gerbicz checks and
retries to catch the rare bad case. The searches found nothing newer that would justify more
bits per word.

Sources: Percival 2003 (ams.org/mcom/2003-72-241/S0025-5718-02-01419-9, PDF at
daemonology.net/papers/fft.pdf); Li & Gallot, arXiv:2209.15623; github.com/patnashev/llr2
(archived, replaced by PRST); mersenneforum "Alternative LLR tester implementation"
(showthread.php?t=25343).

## 2026-10-12 21:29 — Decision: a PRST-compatible CLI with proofs

Goal: ahnum as a drop-in for prst for Fermat tests of k·2^n − 1 with proofs. The slice of
PRST needed: the Fermat test (`EXPR -d -fermat a 3 [-t T]`), `-proof save 512` / `build 512`
/ `cert FILE`, `-time progress 5 coarse`, `-fft info`, `-v`, and SIGTERM checkpointing and
resume in the working directory; LLR (`EXPR -d`) stays with PRST. For k·2^n − 1 with small k PRST takes its
"smooth" path, so the proof is plain Pietrzak (points along the n squarings after a^k), not
Gerbicz–Li. The algorithm and file formats are small: `src/proof.cpp` (~770 lines) and
`framework/file.cpp`, with fingerprint N mod 3417905339, magic 0x9f2b3cd4, 32-bit little-endian
giants and `.md5` sidecar files. Interop is tested against the PRST 14.0.1888 x86 binary
under Rosetta; the source read is 14.1.1903.

Decisions: the CLI reports its own version, `ahnum a.b.c`, rather than imitating PRST's
banner; LLR is handed to PRST for now.

## 2026-10-12 21:43 — ahprst: PRST-compatible proofs, byte for byte

`ahprst` (tools/ahprst.c) implements a slice of PRST's command line (docs/prst-compat.md)
and hands everything else, LLR included, to PRST. The
proof code (src/prstproof.c, src/md5.c) ports PRST 14.1's src/proof.cpp for the smooth path
and framework/file.cpp's format.

How it was checked. PRST 14.0.1888 (x86) ran under Rosetta
to produce reference output: `-proof save 512 keep` leaves all 513 points, so ahnum could
compute the 9 products and the certificate from PRST's own points and compare files. All
matched byte for byte on the first run, including the .md5 sidecars. Then `ahn_prp` gained
a point hook: points are captured as the squaring loop passes them, held until the Gerbicz
check covering them passes, and dropped on a rollback, so a SIGTERM never leaves an
unverified point on disk. With it ahprst's own 513 points for 1203·2^90001 − 1 were
identical to PRST's, and so were its products and certificate (RES64 EC81DC6893506908).
Across the two programs: PRST verifies ahprst's certificate, PRST builds a certificate from
ahprst's points, and ahprst verifies PRST's certificate, all giving the same certificate
RES64. 1203·2^400001 − 1, interrupted with SIGTERM at 25% and resumed, gave a certificate
identical to PRST's (C17686F5F6B4C2CE) and RES64 ACB155E96C137D61, PRST's value.

Details that took reading the source rather than guessing:
- File names use N mod 3417905339, but point and product headers carry
  MD5(fp ‖ "3.n")'s first word, and the certificate header carries the plain fp.
- The challenge is only the first 64 bits of MD5(fp ‖ Y ‖ product): `hash_giants` writes 16
  bytes but then reinitialises the giant to 2 words. It is made odd and pushed past prime
  factors below 1000.
- PRST does not trial-divide by default; its "divisible by" result comes from the Fermat
  test when a divides N.

`tests/prst_compat.sh` (ctest `prst_compat`) holds MD5s of PRST's files for
1203·2^90001 − 1 and checks ahprst against them in 2 s. Still missing then:
PRST's roots-of-unity check during build (a safety check that changes no output), added
before the next entry.

## 2026-10-12 22:00 — ahprst at n = 2,000,003: identical certificates

9001·2^2000003 − 1, with `-proof save 512` then `-proof build 512`, 4 threads each. Both programs give RES64
6B2FB550F09B6CBD and certificate RES64 3B9230016377CE4A; the two .cert files are identical;
PRST verifies ahprst's certificate and ahprst verifies PRST's. ahprst's build ran the new
roots-of-unity check.

Save step, start to finish: ahprst 244.7 s, PRST 364.8 s (1.49×). Not a benchmark: the
machine was not checked for quiet, and PRST runs x86 SSE2 code under Rosetta at 4 threads.
PRST's own "time per op" during a proof save read 0.42 ms against ahprst's 0.12, but PRST's
figure during `-proof save` is not comparable with its plain test, so only wall time
compares. The figure that matters is 14 single-thread proof tests side by side, still to
measure.

## 2026-10-12 22:10 — Deployment target macOS 11

The binaries were marked `minos 27.0`: with no deployment target set, CMake uses the build
machine's macOS, and the loader refuses the binary on anything older. CMakeLists.txt now sets
`CMAKE_OSX_DEPLOYMENT_TARGET` to 11.0, the first macOS for Apple Silicon. A clean build gives
`minos 11.0` for ahprst and ahbench with no availability warnings, ahprst links only
libSystem, and all 11 tests pass. The only linker warnings come from Homebrew's GMP (built
for macOS 15), used by the test tools gmp_ref and test_gmp, which are not shipped.

On other M chips: C is compiled with `-mcpu=apple-m1` and the NEON kernels use ARMv8 FP64
SIMD only. The SME backend is opt-in and checks `hw.optional.arm.FEAT_SME_F64F64` before use.
Without `~/.ahnum/fft_times.txt` (written by `ahbench tune`) ahnum uses its default FFT length
choice and stage order. The macOS 11 build, copied (not rebuilt) to a base M4 Mac mini
(macOS 27, busy with other work), passed `tests/prst_compat.sh` and all 12 known residues
below n = 100,000 with no tuning file. Not yet run on an M1–M3 or on a macOS older than 27.
No M1–M3 is available locally (the other Macs are Intel, where ahprst cannot run and PRST
stays), so `.github/workflows/ci.yml` builds and runs `ctest -LE long` on GitHub's Apple
Silicon runners: macos-14 and macos-15, both M1. That covers an older chip and older macOS
for correctness; the runners are virtualised, so no timings.

## 2026-10-12 22:15 — CI on M1: one test assumed SME

First CI run: on both M1 runners 10 of 11 tests passed, including prst_compat, the known
residues and the GMP checks of the NEON backend; `asm_stages` died with an illegal
instruction. It ran the SME radix-8 kernel unconditionally, and an M1 has no SME. ahnum's
own SME backend already checks `hw.optional.arm.FEAT_SME_F64F64`; the test now does the same
and skips that part. Second run: all 11 tests pass on macOS 14.8.9 (AppleClang 15) and
macOS 15.7.9 (AppleClang 17), both "Apple M1 (Virtual)". The library and ahprst needed no
change for M1 or the older macOS and compilers.

## 2026-10-12 22:30 — Renamed: kite and kite-prst

ahnum is now kite: the library `libkite.a` (header `include/kite.h`), the benchmark command
`kite` (was ahbench) and the PRST-compatible command `kite-prst` (was ahprst), with banner
`kite version 0.1.0`. The name comes from the red kite, a raptor that lives largely on
carrion. The name is free on Homebrew and under primecrunch on GitHub;
several unrelated projects use "Kite", none in prime searching. `kite-prst` keeps a separate
name so it is clear it speaks PRST's command line without being PRST.

Changed with it: `$AHPRST_PRST` → `$KITE_PRST`, checkpoint files `kite_FP.A.ckpt` (tag
`KITECKP1`, so old checkpoints do not resume), tuning file `~/.kite/fft_times.txt`, and the
whole C namespace: 140 `ahn_` names (971 uses in 29 files, including the generated assembly
symbols) became `kite_`, and the 28 `AHN_` names (backend enum, result codes, header guards,
the `AHN_PROF` / `AHN_FFT_TIMES` / `AHN_NO_TAN` style switches) became `KITE_`. The library
exports 152 `kite_` symbols and none with the old prefix; the 11 tests pass, and the GPU
prototypes build against `libkite.a` and give the right residue. This journal's earlier
entries and the patches in `docs/experiments/` keep the old names as written.
The CMake library target is `kitelib` with output name `kite`, because target names must be
unique and the command is `kite`.

## 2026-10-12 22:38 — LLR in kite-prst, matching PRST's Morrison test

`kite-prst EXPR -d` now runs the LLR test itself instead of handing it to PRST.

What PRST computes. Its default test for k·2^n − 1 is the "Morrison (LLR) test"
(`src/morrison.cpp`) with Q = −1, not Riesel's classic u_0 = V_k(P, 1) start. P is the
smallest P ≥ 1 with Jacobi(P² + 4 | N) ≠ 1, and gcd(4P(P² + 4), N) > 1 is reported as a factor
before the test is announced; that is why PRST called 1203·2^90001 − 1 "divisible by 5" (P = 1,
P² + 4 = 5). N is prime iff V_{k·2^(n−1)}(P, −1) ≡ 0, and a composite prints "is not a probable
prime. Have you run Fermat test first? RES64: X" with X the low 64 bits of that V. A Python
prototype reproduced PRST's P and RES64 on 9357·2^20000 − 1 and 9357·2^25000 − 1 before any C
was written.

How kite computes it (`src/llr.c`). V_k by a Lucas ladder on k's bits, then V_2k = V_k² + 2
(k is odd, so Q^k = −1), then n − 2 steps of V² − 2. The constant ±2 needed a new library
operation: `kite_add` / `kite_sub` add two numbers element-wise. That is valid because every
backend stores weighted digits put through linear FFT stages, so the arrays are linear in the
value; the sum's digits are unbalanced until the next square renormalises them. GMP checks on
the ref, c2p and NEON backends (and NEON with 4 threads) pass, including a 30-step V² − 2
chain.

Error handling differs from PRST. Gerbicz checking does not apply to a V² − 2 chain; PRST runs a
Gerbicz–Li check on the Lucas sequence, which is why its "complexity" is about 2n. kite checks
round-off on every step and redoes a block from the last verified state with a longer FFT if
it is too high, and confirms a composite result with a second full run at a longer FFT
length (a third if they disagree). The asymmetry is the point: an arithmetic error turns a
prime's 0 into some non-zero value far more often than it lands exactly on 0, and LLR is
normally run only after a Fermat PRP, so the extra run is almost never paid. The checkpoint
keeps the earlier runs' RES64s, so a confirmation interrupted by SIGTERM resumes inside the
confirmation (tested: interrupted at 62% of run 2, resumed, matched without a third run).

Checked against PRST 14.0.1888: primes 3·2^26459 − 1 and 3·2^41628 − 1; composites
9357·2^20000 − 1 (DC52061161A0DE1C), 9357·2^25000 − 1 (54A27E03D75F2447), 4095·2^70000 − 1
(6399FFF11BF53EC0) and 33·2^99999 − 1 (3FAF838186E8A836), same P each time; 1203·2^90001 − 1
divisible by 5. Five of these are now ctest cases (`llr_*`).

The known prime 9001·2^662695 − 1 (P = 1, FFT 48K) came out prime in 177.2 s on one thread
with the machine under heavy outside load (load average ~46). The primecrunch record for this
number gives its LLR proof as 407 s on the same Mac, also measured under unknown load, so the
two are not a benchmark; a quiet comparison is still to do.

## 2026-10-12 23:05 — kite-prst against PRST, one thread

9001·2^662695 − 1, `-d -fermat a 3 -t 1`, alternating runs on the M4 Pro with the distributed
compute client stopped, each after 60 quiet seconds and thermal pressure 0:

| Run | Program | Time |
|---|---|---|
| 1 | kite-prst | 74.5 s |
| 2 | PRST 14.0.1888 (x86, Rosetta) | 93.5 s |
| 3 | PRST | 93.4 s |
| 4 | kite-prst | 75.4 s |

All four report a probable prime. kite-prst is 1.25× faster on one thread (74.5–75.4 s against
93.4–93.5 s). Run 4 overlapped for about 70 s with a duplicate kite-prst started by mistake
(on another core, same cluster), so it may read about 1% slow. The pre-flight check before it
was held up by WindowServer at 36–43% of a core while the screen was in use; the 20% limit was
set for 14-test runs and is stricter than a one-core test needs.

## 2026-10-16 19:19 — LLR: ready for primecrunch, and a benchmark volunteers can run

How the primecrunch client uses LLR. The server hands out an `llr` job only for a candidate
that the same client has already found to be a Fermat PRP (`ReserveLlrWork.sql`), so nearly
every LLR input is prime. The client runs `prst EXPR -d [-t T]`, reads stdout, and takes a
composite's RES64 as the job's result value; 0 means prime. primecrunch searches only odd k
below 10,000.

A bug in the composite confirmation. kite accepts a 0 (prime) after one run and confirms a
non-zero result with a second run at a longer FFT length, and a third if they disagree. The
verdict was taken from run 0. So a prime whose run 0 went wrong, with runs 1 and 2 agreeing on 0,
would have printed "is not a probable prime ... RES64: 0000000000000000". The verdict and RES64
now come from the agreeing runs. The Lucas ladder for V_k also had no round-off check: the
check after it was reset before the first block. It now has one, and a failed ladder is never
written to the checkpoint.

Fault tests (`tests/test_llr.c`, ctest `llr_faults`, 4 s). New hooks in `kite_llr_opts`
corrupt the value at a chosen doubling in chosen runs, or fail a round-off check once. Covered:
a prime outvoting a bad run 0 (fails on the old code), a composite with run 0 or run 1 wrong,
three different answers (error -3), a round-off redo in the doublings and in the ladder, with
and without a checkpoint, a real too-short FFT (7 redos, right answer), an impossible
round-off limit (gives up with -2), and a stop and resume inside run 0 and inside the
confirmation.

Against PRST 14.0.1888 at production sizes. Eleven composites with P = 1, 3, 5, 7, 8, 12 and 13,
k from 1003 to 987,654,321 and n from 454,049 to 1,999,999: identical RES64s. Seven primes
primecrunch has found (2911·2^333069 − 1, 4299·2^522773 − 1, 4935·2^401257 − 1,
4935·2^401310 − 1, 6369·2^398211 − 1, 8025·2^347103 − 1, 8333·2^339626 − 1): all prime in both.
9357·2^1500007 − 1: both report the factor 5. kite also proved 1011·2^4702339 − 1 prime in
3054 s with 4 threads at FFT length 324K. The same binary gave the same results on a Mac mini
(base M4, 4P+6E, macOS 27), where the non-GMP ctest set also passes. The one miss is k =
3,221,225,471: kite takes k < 2^30 only, for Fermat as well.

On the loaded M4 Pro, kite took about 0.4× PRST's time on the primes (32.9 s against 75.1 s,
71.8 s against 178.9 s) and about the same on composites, which kite runs twice. Not
benchmarks; the machine was at load ~20.

Why PRST's LLR costs 2n. PRST's `morrison.cpp` computes the Lucas U and V sequences together
(`LucasUVMulFast`) when its default `CheckStrong` is on. That makes the test an exponentiation
a Gerbicz–Li check can verify, with 16 check points; carrying U is what doubles the work. With
`CheckStrong` off it runs V alone (`LucasVMulFast`), which is what kite does. The earlier entry
put the 2n down to the Gerbicz–Li check itself; the check costs little, the restructuring
costs 2×. For primecrunch's input (Fermat PRPs) one unchecked run per prime is n against
PRST's ~2n, and a composite costs 2n in both. Gerbicz–Li would win only where errors are
common, since it rolls back a block where kite reruns the whole test. Two runs can agree on a
wrong answer if a fault is systematic, such as a bug in code both runs share; the stronger
safeguard is on the server: an LLR composite after a Fermat PRP is almost certainly a fault
and could be sent to another machine.

Progress now only goes forward, as PRST's does: the first run reports 0–99% and any
confirmation run holds at 99% until the result line. The client would otherwise have shown a
test going from 100% back to 0%. The "confirming" note moved to stderr.

`tools/kite-bench.sh`: a benchmark for volunteers. It waits for 60 quiet seconds (no other
process above 20% of a core, all of them under 50%, thermal pressure nominal). If the machine
stays busy it shows what is busy, grouped by name, with a hint for the primecrunch client,
Spotlight, backups and browsers, and asks the person to quieten it or type go. It then runs
PRST and kite alternately, Fermat (`9357*2^650001-1`, RES64 2AB1256594388021, checked with
both programs) and LLR (`4299*2^522773-1`, prime), one test alone and then one per core, and
writes a Markdown report with both times, the speed ratio and the residues checked against
the known answers. The report gives only load average and other processes' CPU %, no process
names. Testing it on the hot M4 Pro found three hangs or gaps: the cool-down waited for thermal
pressure 0 forever, a bare `wait` would have waited on the background sampler, and the busy
list could be empty at the prompt.

Release workflow (`.github/workflows/release.yml`): a tag vX.Y.Z builds and tests on an M1
runner and publishes the tarball, a copy of `kite-bench.sh` stamped with the version and the
tarball's SHA-256, and SHA256SUMS. Volunteers run `curl -fsSLO .../kite-bench.sh` and `sh
kite-bench.sh`; curl downloads carry no quarantine flag, so the unsigned binary runs without
Gatekeeper prompts, and the script refuses a tarball whose checksum does not match. Tested
locally with a file:// URL, including a tarball with one byte appended.

The script also downloads PRST 14.0 from its GitHub release (`prst-14.0-mac64.7z`, pinned by
SHA-256) unless given a path. macOS's own `tar` unpacks 7-Zip, and the binary inside is
byte-identical to the prst the primecrunch client ships, so every report compares against
the same PRST whether or not the volunteer runs the client. The archive stores prst without
its execute bit, which first showed up as a misleading "needs Rosetta" error; the script now
sets the bit and tests for Rosetta with `arch -x86_64 /usr/bin/true` before blaming it.

## 2026-10-16 20:00 — First full kite-bench run: M4 Pro, quiet

kite 0.1.0 from the GitHub release against PRST 14.0.1888 under Rosetta, run with
`caffeinate -i sh kite-bench.sh`, the distributed compute client stopped. 19 minutes. Other
processes used 28% of one core on average, with one burst to 266%.

| Test | Mode | PRST | kite | kite vs PRST |
|---|---|---|---|---|
| Fermat 9357·2^650001 − 1 | 1 test | 91.3 s | 73.1 s | 1.25× |
| LLR 4299·2^522773 − 1 | 1 test | 131.6 s | 47.9 s | 2.75× |
| Fermat | 14 tests | 145.1 s, 345/h | 122.5 s, 410/h | 1.19× on throughput |
| LLR | 14 tests | 212.6 s, 237/h | 76.4 s, 655/h | 2.76× on throughput |

Every residue matched the known answer in both programs. The one-test Fermat ratio is the
same 1.25× measured by hand on 12 October (9001·2^662695 − 1), so the script agrees with the
manual method. Under full load kite's per-test Fermat time grows 1.68× against PRST's 1.59×,
so its lead drops from 1.25× to 1.19×: the memory and power limits in priority 1 of the speed
file cost kite more than PRST. LLR keeps its lead under load: one Mac running kite does the LLR
work of nearly three running PRST, from half the work per prime (V alone against U and V)
times the faster squaring.

The report said "quiet machine: peak load average 59.7". The load average counts the
benchmark's own 14 tests, so it says nothing about outside noise; the report now gives the
load average from before the runs and the outside-CPU figures from during them. The time
estimate is now 20–40 minutes rather than 30. While the user's run waited in its quiet
check, the status line stayed at "0s of 60s" with no reason given; the next version names
the busiest process while it waits. My own test copy of the script, started at the same
time, briefly ran PRST alongside it before I stopped it; that could only have delayed the
quiet check, not the measured runs.

## 2026-10-16 20:29 — kite-bench 0.1.1 rerun: the figures repeat

Same M4 Pro, quiet (load 1.06 before the runs, other processes 41% of one core on average
during them). Fermat one test 90.9 s / 73.0 s (1.25×), LLR one test 132.6 s / 48.4 s (2.74×),
Fermat 14 tests 341/h / 413/h, LLR 14 tests 234/h / 655/h. Every figure is within about 1% of
the first run, so one run per machine is enough for the README table. The second report
replaced the first (same chip and date in the file name); a repeat run now gets -2, -3, ....
