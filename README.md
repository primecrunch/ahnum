# kite

Fast modular squaring of large integers modulo Riesel numbers N = k·2^n − 1 on Apple Silicon,
built for Fermat probable-prime testing. Hot kernels are generated AArch64 NEON assembly; a
thin C layer handles setup, threading, error checking and the API.

## Build and test

```sh
cmake -S . -B build && cmake --build build -j
cd build && ctest -LE long        # quick suite (GMP comparisons, asm kernel tests, known residues)
ctest -L long                     # adds 9357*2^369717-1 (primecrunch residue)
```

GMP (Homebrew) is needed only for the tests and `gmp_ref`.

## Benchmark CLI

```sh
build/kite prp "9357*2^369717-1" --expect F5FDA01218F20439
# 9357*2^369717-1 is composite  RES64 F5FDA01218F20439  FFT 28672  26.2 s (70.9 us/iter) ...
# MATCH expected F5FDA01218F20439

build/kite prp 69 24612729 --threads 8 --ckpt 69.ckpt   # record-size test, resumable
build/kite time 9357 369717                             # us per squaring
build/kite throughput 9357 369717 --configs 10x1,5x2,2x5 # tests x threads splits
build/kite tune 8192 196608 --out ~/.kite/fft_times.txt # measure FFT lengths
```

RES64 is the low 64 bits of 3^(N−1) mod N in upper-case hex (primecrunch convention). A PRP has
RES64 `0000000000000001`. `--expect` makes the exit status reflect the comparison.

Options: `--threads T`, `--fft L` (force a length), `--order 0..3` (force an FFT stage order),
`--backend neon|c2p|ref`, `--base B`,
`--ckpt FILE` / `--ckpt-secs S` (checkpoint/resume), `--block B` (Gerbicz block),
`--fault ITER` (inject a digit error to exercise the Gerbicz check).

## PRST-compatible CLI (`kite-prst`)

`build/kite-prst` accepts a subset of PRST's command line for k·2^n − 1: Fermat tests, `-proof save/build/cert` and progress output, with checkpoints on SIGTERM.
Other tests, LLR included, are passed to PRST (`$KITE_PRST`, or `prst` next to
`kite-prst`). Proof points, products and certificates are byte-identical to PRST 14.0's, and each
program verifies the other's certificates; see `docs/prst-compat.md`.

```sh
build/kite-prst "9001*2^2000003-1" -d -fermat a 3 -proof save 512
build/kite-prst "9001*2^2000003-1" -d -fermat a 3 -proof build 512
build/kite-prst "9001*2^2000003-1" -d -proof cert prst_<fingerprint>.cert
```

PRST is by Pavel Atnashev (https://github.com/patnashev/prst). The proof code in
`src/prstproof.c` reimplements PRST's proof construction and file format so the two
interoperate.

## Reference benchmark

```sh
tools/bench.sh
# 9001*2^662695-1  ok  74.6s  reference 195s (primecrunch fermat job on wolf (darwin/arm64))  speedup 2.61x
```

`tests/benchmarks.csv` lists numbers with known residues and reference times. The script runs
the full Gerbicz-checked test, checks the residue, and reports the speedup.

### Speed history

9001·2^662695−1, full Gerbicz-checked Fermat test on a quiet M4 Pro. µs/squaring is the full
test time divided by n. Throughput is complete concurrent single-thread tests, timed end to end:
10 on the P-cores, and 14 on all cores (P + E; "—" where it was not measured). Each row adds
one improvement; details are in `journal.md`. Each figure is one round of tests (~2 minutes)
started on a cool machine. The machine is a 16" MacBook Pro: under sustained load its clocks drop
~10% as thermal pressure rises, and 14 tests sustain ~372 tests/h at `3631fc5` (25-minute run,
2026-10-12). The first 14-test measurement at `3631fc5` read 408 tests/h with `duetexpertd`
holding a core; the quiet rerun gave 423.

| Version | Commit | FFT | One core | µs/squaring | 10-test throughput | 14-test throughput |
|---|---|---|---|---|---|---|
| prst (gwnum via Rosetta 2), for reference | — | 49152 | 90.2 s | 136 | 283–307 tests/h | 329 tests/h |
| First NEON build | `10b8039` | 51200 | 91.6 s | 138 | 298 tests/h | 321 tests/h |
| Exact fixed-point IBDWT weights (shorter FFT) | `9dca186` | 49152 | 85.7 s | 129 | 295–313 tests/h | — |
| Row transpose fused into the row FFT | `3780a4b` | 49152 | 80.3 s | 121 | 336–340 tests/h | — |
| In-place normaliser, tangent DIT twiddles, tuned stage order | `3631fc5` | 49152 | 74.6 s | 113 | 371 tests/h | 423 tests/h |

GPU prototype (`gpu/`, on hold): `gpu/gpu_prp.m` runs complete Gerbicz-checked tests on the GPU,
173 tests/hour alone for 9001·2^662695−1 with 64 in flight. On this MacBook Pro, running it
alongside the CPU tests loses throughput: within a minute macOS caps the P-cores at 1800 MHz and
~5 W to give the GPU its power, and the combination sustains ~244 tests/hour against ~372 for the
CPU alone. The +22% from 30 s squaring-rate samples measured the first minute, before the cap.
It may still pay on a desktop with a bigger power budget.

## Verified residues from primecrunch

```sh
tools/primecrunch_fetch.py 9357:369717                       # print k,n,res64,source
tools/primecrunch_fetch.py --k 9357 --n-from 369500 --n-to 369699 --append tests/known_residues.csv
tests/run_known.sh build/kite tests/known_residues.csv 1000000
```

The fetcher uses the public `GET /v2/k/{k}/n/{n}` on `https://api.primecrunch.com` and uses
completed `fermat` jobs only. The `servers` entry in the primecrunch OpenAPI spec
(`https://primecrunch.com`) serves the web app, not the API.

## Library API (`include/kite.h`)

```c
kite_prp_opts o = {0};               // o.fft.threads, o.checkpoint, o.progress, ...
kite_prp_result r;
if (kite_prp(9357, 369717, 3, &o, &r) == 0)
    printf("%016llX %s\n", r.res64, r.is_prp ? "PRP" : "composite");
```

Lower level: `kite_ctx_new(k, n, -1, &opts)`, `kite_num_new`, `kite_set_u64` / `kite_set_limbs`,
`kite_square`, `kite_square_n`, `kite_mul`, `kite_mul_small`, `kite_get_limbs`, `kite_res64`,
`kite_max_roundoff`.

## How it works

**Arithmetic.** A weighted FFT (generalised IBDWT, Percival 2003). k is folded into the
irrational base: with T = n + log2 k, word j starts at bit s_j = ⌈jT/L⌉ and has weight
2^(s_j − jT/L) ∈ [1, 2). The weighted cyclic convolution then yields integers plus wrapped parts
divided by k, so outputs are multiplied by k before rounding. Numbers are therefore stored as
x/k ≡ x·2^n (mod N) internally. The carry out above 2^n is "folded": H·2^n with H = qk + r
becomes q + r·2^n. The cost of k is about 0.5·log2 k bits per FFT word, as with gwnum.

**FFT.** A real length-L convolution via a complex FFT of H = L/2 points, four-step H = R×C:
- **Pass 1** works on groups of 8 columns. It runs the inverse column FFT, normalisation
  (unweight, round, carry, reweight), then the forward column FFT. The four-step twiddle is fused
  into the outer stages.
- **Pass 2** processes row pairs (p, p′) with k_r′ = −k_r, transposed so both rows sit in the
  two NEON lanes. It runs the row FFT, the real-split squaring (partners are lane-swapped
  mirror elements) and the inverse row FFT.

Stages are radix 2/3/4/5/7/8, so lengths are 2^a·3^b·5^c·7^d·64.

**Kernels** (`gen/gen_kernels.py` → `build/asm/*.S`):
- DIF/DIT stage kernels per radix, with by-element twiddle FMAs and four-step-twiddle variants.
  DIT stages of radix 2/3/4/8 use a tangent form, w = c(1 + iτ): two FMAs per twiddle, with
  c folded into the butterfly's own FMAs (`KITE_NO_TAN=1` selects the plain complex multiply).
- Normalisers: compressed-weight and table-driven variants. The table variants work each digit
  in place from a {osc, 1/base, base·w} record, and the 8-row version keeps four carry chains
  in flight to hide FP latency.
- Row transposes and the real-split kernel.

At 4.49 GHz all of them run at about 4 FP instructions per cycle in isolation, the limit of the
4 NEON pipes. In the full squaring the passes that load from L2 run lower; `KITE_PROF=1` prints a
per-phase time split, and `tools/count_ops.py build/asm` counts each kernel's instructions.

**FFT length** comes from a fitted round-off model. Measured: bpw_max ≈ 21.44 −
0.305·(log2 L − 12) − 0.525·log2 k, less a 0.45 margin. Among the safe lengths, the fastest is
taken from `~/.kite/fft_times.txt` (or `$KITE_FFT_TIMES`); without that file a heuristic is
used. Each line is "L us order": `kite tune` times the four stage orders (column and row
radices in the default order or ascending) and records the fastest, since which is faster
depends on the length. `tune/fft_times_m4pro.txt` holds measurements for this machine.

**Error checking** (`src/prp.c`):
- A Gerbicz check every B·M squarings (B ≈ √n, M ≈ B/4), with rollback to the last verified
  state.
- Round-off is sampled every 4th squaring; above 0.40 the test restarts from the verified state
  with a ≥4% longer FFT.
- Checkpoints are written atomically, with a checksum.
- The final < B squarings are computed twice and compared.

**Threading.** A fork-join pool (QoS user-interactive, time-bounded spin then sleep). Pass 2
splits row pairs across threads. Pass 1 splits column groups; each thread defers its first group
and fixes it up with the previous range's carries.

## Measured (M4 Pro, 10P+4E, quiet machine)

| number | FFT | config | time/squaring | per test |
|---|---|---|---|---|
| 9357·2^369717−1 | 26880 | 1 thread | 64.3 µs | 24 s |
| 9001·2^662695−1 | 49152 | 1 thread | 113 µs | 74.6 s |
| 9001·2^662695−1 | 49152 | 10 tests × 1 thread | — | 371 tests/hour (97 s each) |
| 9001·2^662695−1 | 49152 | 14 tests × 1 thread (P+E), first round | — | 423 tests/hour (119 s each) |
| 9001·2^662695−1 | 49152 | 14 tests × 1 thread (P+E), sustained 25 min | — | ~372 tests/hour (~135 s each) |
| 9001·2^662695−1 | 49152 CPU, 32768 NTT on GPU | 13 tests × 1 thread + GPU prototype, sustained | — | ~244 tests/hour (85 CPU + 159 GPU) |
| 9001·2^662695−1 | 32768 NTT on GPU | GPU prototype alone, 64 numbers in flight | 30.6 µs | 173 tests/hour |
| 9357·2^369717−1 | 26880 | 10 tests × 1 thread | 74 µs each | 1314 tests/hour |
| 9357·2^369717−1 | 26880 | 14 tests × 1 thread (P+E) | 97 µs each | 1405 tests/hour |
| 69·2^24612729−1 | 1638400 | 1 thread | 6.5 ms | 45 h |
| 69·2^24612729−1 | 1638400 | 10 threads | 0.77 ms | 5.3 h |

The 9001·2^662695−1 throughput rows are complete tests with checked residues (2026-10-12). The
GPU path (`gpu/`) is a prototype on hold: a complete squaring mod k·2^n−1 (4-prime integer NTT,
Garner CRT, Montgomery reduction with R = 2^n) with Gerbicz checks and checkpoints, at one
transform length (32768).

At n ≈ 370K, splitting a test across threads loses throughput (5×2 gives 892 tests/hour, 1×10
gives 510), so run one test per core. For record sizes, run one test across all P-cores.
`kite throughput` measures the best split for a given size.

## SME (M4) — measured, not used by default

`--backend sme` runs the pass-1 radix-8 column stages on the SME unit. Each stage becomes 8×8
complex matrix products (FP64 FMOPA), with the twiddles folded into per-j matrices
(`kite_sme_r8`). It is correct (GMP and primecrunch tests pass), but it is slower end to end:

- The FMOPA rate is 492 GFLOPS from one core. A radix-8 stage needs 32 outer products per
  8-point × 8-lane item, which gives 0.85 cycles/point against 1.3 for NEON in isolation.
- The SME unit works from L2. Handing a group between NEON (normalisation) and SME (FFT) costs
  about 400 ns per 8 KB group each way, about 28 cycles per cache line. Normalising in
  streaming mode is not an option because streaming-SVE vector arithmetic runs at about 15
  GFLOPS.
- Result: 96 µs per squaring against 72 µs for NEON at FFT 28672 (35.6 s against 26.7 s for
  9357·2^369717−1). The unit is also shared by the 5 cores of a cluster, so throughput mode
  cannot gain from it either.

The probes are in `tools/sme_probe.*`, plus `kbench 4.49 sme` and `kbench 4.49 handoff`.

## Layout

```
include/kite.h       public API
src/ctx.c             layout, weights, carries/fold, FFT length choice, conversions
src/neon.c            NEON backend (two-pass, threads); `sme` variant of it
src/prp.c             Fermat PRP driver: Gerbicz, round-off fallback, checkpoints
src/threads.c         fork-join pool
src/fast2p.c          same algorithm in portable C (independent cross-check)
src/ref_fft.c         scalar reference backend (oracle, tiny sizes)
gen/gen_kernels.py    assembly generator
tools/                kite, gmp_ref, xcheck, kbench, sweep_err, fit_bpw.py, sme_probe
tests/                GMP comparisons, stage-kernel tests, known residues
```
