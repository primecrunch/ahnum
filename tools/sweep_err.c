// Measure max round-off vs bits/word for a fixed FFT length.
// Usage: sweep_err backend L k bpw_lo bpw_hi step [iters]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ahnum.h"

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s ref|neon|sme L k bpw_lo bpw_hi step [iters]\n", argv[0]);
        return 2;
    }
    ahn_opts o = {0};
    o.backend = !strcmp(argv[1], "neon") ? AHN_BACKEND_NEON
              : !strcmp(argv[1], "sme")  ? AHN_BACKEND_SME
              : !strcmp(argv[1], "c2p")  ? AHN_BACKEND_C2P
                                         : AHN_BACKEND_REF;
    o.fft_len = (uint32_t)atoi(argv[2]);
    uint64_t k = strtoull(argv[3], 0, 10);
    double lo = atof(argv[4]), hi = atof(argv[5]), step = atof(argv[6]);
    int iters = argc > 7 ? atoi(argv[7]) : 200;
    for (double b = lo; b <= hi + 1e-9; b += step) {
        uint64_t n = (uint64_t)(b * o.fft_len);
        ahn_ctx *ctx = ahn_ctx_new(k, n, -1, &o);
        if (!ctx) {
            printf("bpw=%.2f: no ctx\n", b);
            continue;
        }
        ahn_num *x = ahn_num_new(ctx);
        ahn_set_u64(ctx, x, 3);
        ahn_square_n(ctx, x, 64);  // fill all words with pseudo-random data
        ahn_reset_roundoff(ctx);
        ahn_square_n(ctx, x, iters);
        printf("L=%u k=%llu bpw=%.2f n=%llu maxerr=%.4f\n", o.fft_len, (unsigned long long)k, b,
               (unsigned long long)n, ahn_max_roundoff(ctx));
        fflush(stdout);
        ahn_num_free(ctx, x);
        ahn_ctx_free(ctx);
    }
    return 0;
}
