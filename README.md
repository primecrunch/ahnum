# ahnum

Fast modular squaring of large integers modulo Riesel numbers N = k·2^n − 1 on Apple Silicon,
built for Fermat probable-prime testing. Hot kernels are generated AArch64 NEON assembly; a
thin C layer handles setup, threading, error checking and the API.

## Build and test

```sh
cmake -S . -B build && cmake --build build -j
cd build && ctest -LE long        # quick suite (GMP comparisons, asm kernel tests, known residues)
ctest -L long                     # adds 9357*2^369717-1 (PrimeCrunch residue)
```

GMP (Homebrew) is needed only for the tests and `gmp_ref`.

## Benchmark CLI

```sh
build/ahbench prp "9357*2^369717-1" --expect F5FDA01218F20439
# 9357*2^369717-1 is composite  RES64 F5FDA01218F20439  FFT 28672  26.2 s (70.9 us/iter) ...
# MATCH expected F5FDA01218F20439

build/ahbench prp 69 24612729 --threads 8 --ckpt 69.ckpt   # record-size test, resumable
build/ahbench time 9357 369717                             # us per squaring
build/ahbench throughput 9357 369717 --configs 10x1,5x2,2x5 # tests x threads splits
build/ahbench tune 8192 196608 --out ~/.ahnum/fft_times.txt # measure FFT lengths
```

RES64 is the low 64 bits of 3^(N−1) mod N in upper-case hex (PrimeCrunch convention). A PRP has
RES64 `0000000000000001`. `--expect` makes the exit status reflect the comparison.

Options: `--threads T`, `--fft L` (force a length), `--backend neon|c2p|ref`, `--base B`,
`--ckpt FILE` / `--ckpt-secs S` (checkpoint/resume), `--block B` (Gerbicz block),
`--fault ITER` (inject a digit error to exercise the Gerbicz check).

## Verified residues from PrimeCrunch

```sh
tools/primecrunch_fetch.py 9357:369717                       # print k,n,res64,source
tools/primecrunch_fetch.py --k 9357 --n-from 369500 --n-to 369699 --append tests/known_residues.csv
tests/run_known.sh build/ahbench tests/known_residues.csv 1000000
```

The fetcher uses the public `GET /v2/k/{k}/n/{n}` on `https://api.primecrunch.com` and uses
completed `fermat` jobs only. The `servers` entry in `docs/primecrunch-openapi.yaml`
(`https://primecrunch.com`) serves the web app, not the API.

## Library API (`include/ahnum.h`)

```c
ahn_prp_opts o = {0};               // o.fft.threads, o.checkpoint, o.progress, ...
ahn_prp_result r;
if (ahn_prp(9357, 369717, 3, &o, &r) == 0)
    printf("%016llX %s\n", r.res64, r.is_prp ? "PRP" : "composite");
```

Lower level: `ahn_ctx_new(k, n, -1, &opts)`, `ahn_num_new`, `ahn_set_u64` / `ahn_set_limbs`,
`ahn_square`, `ahn_square_n`, `ahn_mul`, `ahn_mul_small`, `ahn_get_limbs`, `ahn_res64`,
`ahn_max_roundoff`.

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
- Normalisers: compressed-weight and table-driven variants. The 4-row version interleaves two
  carry chains to hide FP latency.
- Row transposes and the real-split kernel.

At 4.49 GHz all of them run at about 4 FP instructions per cycle, the limit of the 4 NEON pipes.

**FFT length** comes from a fitted round-off model. Measured: bpw_max ≈ 21.44 −
0.305·(log2 L − 12) − 0.525·log2 k, less a 0.45 margin. Among the safe lengths, the fastest is
taken from `~/.ahnum/fft_times.txt` (or `$AHN_FFT_TIMES`); without that file a heuristic is
used. `tune/fft_times_m4pro.txt` holds measurements for this machine.

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

## Measured (M4 Pro, 10P+4E)

| number | FFT | config | time/squaring | per test |
|---|---|---|---|---|
| 9357·2^369717−1 | 28672 | 1 thread | 70.9 µs | 26.2 s |
| 9357·2^369717−1 | 28224 | 10 tests × 1 thread | ~100 µs each | ~950–1200 tests/hour |
| 69·2^24612729−1 | 1638400 | 8 threads | 1.28 ms | ~8.8 h |

The best tests × threads split depends on the FFT size and cache fit. Measure it with
`ahbench throughput`.

## Layout

```
include/ahnum.h       public API
src/ctx.c             layout, weights, carries/fold, FFT length choice, conversions
src/neon.c            NEON backend (two-pass, threads)
src/prp.c             Fermat PRP driver: Gerbicz, round-off fallback, checkpoints
src/threads.c         fork-join pool
src/fast2p.c          same algorithm in portable C (independent cross-check)
src/ref_fft.c         scalar reference backend (oracle, tiny sizes)
gen/gen_kernels.py    assembly generator
tools/                ahbench, gmp_ref, xcheck, kbench, sweep_err, fit_bpw.py, sme_probe
tests/                GMP comparisons, stage-kernel tests, known residues
```
