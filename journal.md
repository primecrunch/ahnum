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
- Gerbicz plus round-off checking. Verified residues come from the PrimeCrunch API.
- Throughput matters most. The tests × threads split has to be measured per FFT size.

## 10:48 — Pinning the residue convention

There are several plausible "Fermat residue" definitions: 3^(N−1), 3^(N+1), the half
exponents, and so on. A GMP oracle printing all of them took 38 minutes for one 370K-bit
number. The project's own first FFT finished sooner, and it showed that PrimeCrunch's
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

## 12:35 — PrimeCrunch residues

The API is served from `https://api.primecrunch.com`. The OpenAPI `servers` entry,
`https://primecrunch.com`, returns the web app's HTML for `/v2/...`.

GET `/v2/k/{k}/n/{n}` is public and lists jobs with a `residual`. Only `fermat` jobs are used,
because LLR residues are a different quantity. No endpoint lists candidates, so
`tools/primecrunch_fetch.py` takes (k, n) pairs or does a rate-limited scan; sieved-out n
return 404.

From n = 369500…369699, 13 residues for k = 9357 were fetched, and ahbench matched all 13
(10 tests in parallel). The PrimeCrunch record for 9357·2^369717−1 shows the original result
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

PrimeCrunch records this known prime's Fermat job at 195 s on wolf (this M4 Pro, darwin/arm64
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
