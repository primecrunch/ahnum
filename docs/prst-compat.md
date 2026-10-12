# PRST compatibility (`ahprst`)

PRST is by Pavel Atnashev (https://github.com/patnashev/prst). ahnum's proof code
(`src/prstproof.c`) reimplements PRST's proof construction and file format, following its
`src/proof.cpp` and the `arithmetic` framework's `file.cpp`, so that certificates made by
either program verify with the other.

`ahprst` stands in for `prst` for Fermat tests and proofs of k·2^n − 1. This file lists the
arguments, output and files it reproduces, checked against PRST 14.0.1888 (x86, under
Rosetta) and the PRST 14.1 source.

Only numbers k·2^n − 1 with small k are in scope. For those PRST's Fermat test takes its
"smooth" path (`fermat.cpp`: b = 2, log2 k < n/4): x_0 = a^k, n squarings, then the tail
a^−2, so RES64 is the low 64 bits of a^(N−1), the same as ahnum's.

## Command lines

| Use | Arguments |
|---|---|
| Fermat test | `EXPR -d -fermat a 3 [-t T]` |
| Fermat test with proof points | `EXPR -d -fermat a 3 [-t T] -proof save 512` |
| Certificate build | `EXPR -d -fermat a 3 [-t T] -proof build 512` |
| Certificate check | `EXPR -d [-t T] -proof cert verify.cert` |
| LLR (probable primes) | `EXPR -d [-t T]` — handed to PRST |
| Benchmark | Fermat test plus `-time progress 5 coarse` |
| FFT probe | `EXPR -fermat a 3 -fft info` |
| Version | `-v` |
| Retry after a tail error | adds `-cpu SSE2`; accepted and ignored |

EXPR is `K*2^N-1`. Everything runs in the working directory: checkpoints, proof files,
`result.txt`.

## Output (stdout)

Lines end with `\n`, except progress readings, which end with ` \r` and are followed by 70
spaces and `\r` before the next line. Prefix = `EXPR ` on per-test lines.

    Using FFT length 6K.                                   (ahprst: names its own transform)
    Fermat probabilistic test of EXPR, a = 3, complexity = C.
    EXPR Gerbicz check enabled, L2 = A*B.
    EXPR restarting at P%.                                 (resumed from a checkpoint)
    EXPR Saving 512 proof points.                          (save)
    EXPR P% done, time per op: T ms. \r
    EXPR is not prime. RES64: XXXXXXXXXXXXXXXX, time: S.s s.
    EXPR is a probable prime. Time: S.s s.
    EXPR is not prime, divisible by D.                     (trial division; no RES64)
    EXPR compressed 512 points to 9 products, time: S.s s. (save)
    EXPR raw certificate RES64: XXXXXXXXXXXXXXXX.          (save)
    Factorizing N-1 for roots of unity check, time: S.s s. (build)
    EXPR Building certificate from 9 products.             (build)
    EXPR a^k is correct. Time: S.s s.                      (build)
    EXPR restarting at 100.0%.                             (build)
    EXPR certificate RES64: XXXXXXXXXXXXXXXX, time: S.s s. (build and cert)
    Verifying certificate of EXPR, complexity = C.         (cert)
    EXPR max roundoff check enabled.                       (cert)

Patterns a caller can rely on: progress `(\d+(\.\d+)?)% (done|total), time per op: [0-9.]+ ms`,
`restarting at P%.`, `is not prime ... R??64: HEX`, `is prime[!.]`, `is a probable prime`,
`is not prime, divisible by D`, `raw certificate RES64:`, `certificate RES64:`,
`Fermat probabilistic test of .*, a = (\d+)`, `complexity = (\d+)`,
`Using (.*?)\s*FFT length (\d+)([KM]?)`.

`-v` prints `ahnum version a.b.c`, not PRST's `PRST version X, GWnum library version Y`.

## result.txt

One line appended per result, with whole seconds:

    EXPR is not prime. RES64: XXXXXXXXXXXXXXXX, time: 1 s.
    EXPR raw certificate RES64: XXXXXXXXXXXXXXXX, time: 0 s.
    EXPR certificate RES64: XXXXXXXXXXXXXXXX, time: 0 s.



## Process behaviour

- SIGTERM: write a checkpoint, print `Terminating...`, exit 1. The next run with the same
  arguments resumes and prints `restarting at P%.`. Checkpoint files are removed when the test
  completes.
- Exit 0 on a result.

## Proof files

All files are written with an `.md5` sidecar holding the lowercase hex MD5 of the file.

Header: `u32 0x9f2b3cd4`, `u32 appid | type << 16 | version << 24` (appid 4, version 0), `u32
fingerprint`, then the state. TaskState: `i32 iteration`. Giant: `i32 word count` (negative
for negative values), then that many little-endian 32-bit words, no leading zero words.

| File | Type | State |
|---|---|---|
| `prst_FP.proof.I` (I = 0..512) | 1 (value) | iteration = point position; giant x_pos (point 0: a^k, unreduced) |
| `prst_FP.prod.I` (I = 0..8) | 3 (product) | iteration = I; giant product |
| `prst_FP.cert` | 4 (certificate) | iteration = M; giant X; giant a_power = 0; giant a_base = 0 |

Fingerprints: `fp(N) = N mod 3417905339`; `unique(fp, s) = first u32 of MD5(u32 fp ‖ s)`.
File names use `fp(N)` (`prst_1107603830.*` for 1203·2^90001 − 1). In the headers, the
certificate carries `fp(N)` and the point and product files carry `unique(fp(N), "3." + n)`
(a = 3, last point position n); both checked against PRST 14.0 output.

Save deletes points 1..511 unless `keep`; build deletes points 0 and 512 and the products.
Point positions, the product tree, the challenge hashes (`MD5(u32 fp(N) ‖ Y words ‖ product
words)`, made odd and free of prime factors below 1000) and the certificate exponentiations
follow `src/proof.cpp` exactly; `ahnum/src/proof.c` cites the lines it ports.

## ahprst specifics and gaps

- `-v` prints `ahnum version a.b.c` (`AHNUM_VERSION` in `include/ahnum.h`).
- The FFT line names ahnum's transform, e.g. `Using NEON FFT length 48K.`; lengths that are
  not a multiple of 1024 print in full (`26880`), which the FFT pattern above accepts.
- `complexity` is n + n/B (squarings plus the Gerbicz multiplications), so complexity × time
  per op is the length of the test.
- Checkpoints are ahnum's own format, `ahprst_FP.A.ckpt`, written every 300 s and on SIGTERM.
  A PRST checkpoint left in the directory is ignored; the test starts again.
- Proof points are written only after the Gerbicz check covering them passes, so an
  interrupted run never leaves an unverified point on disk.
- Anything outside the table above, LLR included, is handed to PRST: `$AHPRST_PRST`, or
  `prst` next to the `ahprst` binary.
- Not implemented yet: the roots-of-unity check PRST runs during `-proof build` (a safety
  check on the residue; it does not change any file or RES64), Gerbicz–Li certificates
  (only made for non-smooth tests, out of scope here), and resuming the
  product step part-way (it is redone; it takes seconds).

- Progress always reads `P% done`. PRST writes `P% stage / Q% total` during a proof save;
  the progress pattern above accepts both.

Tested (`tests/prst_compat.sh`, and by hand against PRST 14.0.1888 under Rosetta):
1203·2^90001 − 1 and 1203·2^400001 − 1 (interrupted with SIGTERM and resumed) give point,
product and certificate files identical to PRST's; PRST verifies ahprst's certificates and
builds from its points; ahprst verifies PRST's certificates. 9001·2^2000003 − 1, with 4
threads each: same RES64 (6B2FB550F09B6CBD), identical
certificate files (certificate RES64 3B9230016377CE4A), and each program verifies the
other's certificate.
