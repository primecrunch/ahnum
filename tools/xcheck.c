// Cross-check two backends/configurations: run the same squaring chain from
// 3 and compare the full residues.
// Usage: xcheck K N ITERS FFT  BACKEND_A THREADS_A  BACKEND_B THREADS_B
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ahnum.h"

static ahn_backend parse_backend(const char *s) {
    if (!strcmp(s, "ref")) return AHN_BACKEND_REF;
    if (!strcmp(s, "c2p")) return AHN_BACKEND_C2P;
    return AHN_BACKEND_NEON;
}

static uint64_t *run(uint64_t k, uint64_t n, uint64_t iters, uint32_t L, ahn_backend b,
                     uint32_t th, size_t *nl, double *err) {
    ahn_opts o = {b, L, th};
    ahn_ctx *c = ahn_ctx_new(k, n, -1, &o);
    if (!c) {
        fprintf(stderr, "no context\n");
        exit(1);
    }
    ahn_num *x = ahn_num_new(c), *y = ahn_num_new(c);
    ahn_set_u64(c, x, 3);
    ahn_set_u64(c, y, 7);
    for (uint64_t i = 0; i < iters; i++) {
        ahn_square(c, x);
        if (i % 16 == 15) ahn_mul(c, x, y);  // exercise multiplication too
    }
    *nl = ahn_limbs(c);
    uint64_t *l = malloc(*nl * 8);
    ahn_get_limbs(c, x, l);
    *err = ahn_max_roundoff(c);
    ahn_num_free(c, x);
    ahn_num_free(c, y);
    ahn_ctx_free(c);
    return l;
}

int main(int argc, char **argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s K N ITERS FFT BACKEND_A THREADS_A BACKEND_B THREADS_B\n", argv[0]);
        return 2;
    }
    uint64_t k = strtoull(argv[1], 0, 10), n = strtoull(argv[2], 0, 10);
    uint64_t iters = strtoull(argv[3], 0, 10);
    uint32_t L = (uint32_t)atoi(argv[4]);
    size_t na, nb;
    double ea, eb;
    uint64_t *a = run(k, n, iters, L, parse_backend(argv[5]), (uint32_t)atoi(argv[6]), &na, &ea);
    uint64_t *b = run(k, n, iters, L, parse_backend(argv[7]), (uint32_t)atoi(argv[8]), &nb, &eb);
    int same = na == nb && !memcmp(a, b, na * 8);
    printf("%s  %016" PRIX64 " vs %016" PRIX64 "  maxerr %.4f / %.4f\n", same ? "SAME" : "DIFFERENT",
           a[0], b[0], ea, eb);
    return !same;
}
