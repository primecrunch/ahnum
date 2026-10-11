// Compare ahnum arithmetic against GMP for random (k, n) and random inputs.
// Usage: test_gmp [ref|neon|sme] [trials]
#include <gmp.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ahnum.h"

static int ahn_fft_len_ok_pub(uint32_t L) {
    if (L < 16 || (L & 1)) return 0;
    while (L % 2 == 0) L /= 2;
    while (L % 3 == 0) L /= 3;
    while (L % 5 == 0) L /= 5;
    while (L % 7 == 0) L /= 7;
    return L == 1;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static void to_mpz(mpz_t r, const uint64_t *l, size_t n) {
    mpz_import(r, n, -1, sizeof(uint64_t), 0, 0, l);
}

static void from_mpz(uint64_t *l, size_t n, const mpz_t x) {
    size_t cnt = 0;
    memset(l, 0, n * sizeof(uint64_t));
    mpz_export(l, &cnt, -1, sizeof(uint64_t), 0, 0, x);
}

static int check(ahn_ctx *ctx, const ahn_num *x, const mpz_t want, const char *what,
                 uint64_t k, uint64_t n) {
    size_t nl = ahn_limbs(ctx);
    uint64_t *l = malloc(nl * sizeof(uint64_t));
    ahn_get_limbs(ctx, x, l);
    mpz_t got;
    mpz_init(got);
    to_mpz(got, l, nl);
    int bad = mpz_cmp(got, want) != 0;
    if (bad) {
        gmp_printf("FAIL %s k=%" PRIu64 " n=%" PRIu64 " L=%u\n  got  %Zx\n  want %Zx\n", what, k,
                   n, ahn_fft_len(ctx), got, want);
    }
    mpz_clear(got);
    free(l);
    return bad;
}

int main(int argc, char **argv) {
    ahn_opts o = {0};
    if (argc > 1 && !strcmp(argv[1], "ref")) o.backend = AHN_BACKEND_REF;
    if (argc > 1 && !strcmp(argv[1], "neon")) o.backend = AHN_BACKEND_NEON;
    if (argc > 1 && !strcmp(argv[1], "sme")) o.backend = AHN_BACKEND_SME;
    if (argc > 1 && !strcmp(argv[1], "c2p")) o.backend = AHN_BACKEND_C2P;
    int trials = argc > 2 ? atoi(argv[2]) : 40;
    if (argc > 3) o.threads = (uint32_t)atoi(argv[3]);
    int fails = 0;
    mpz_t N, a, b, t;
    mpz_inits(N, a, b, t, NULL);
    for (int tr = 0; tr < trials; tr++) {
        uint64_t k = (rng() % 10000) | 1;
        if (tr == 0) k = 1;
        if (tr == 1) k = 9357;
        uint64_t n = 600 + rng() % (tr < trials / 2 ? 3000 : 60000);
        // Odd trials force a low bits/word so k spans several top words.
        ahn_opts oo = o;
        if (tr & 1) {
            uint32_t L = (uint32_t)(n / (6 + rng() % 6));
            L = (L + 63) & ~63u;
            while (!ahn_fft_len_ok_pub(L)) L += 64;
            oo.fft_len = L;
        }
        ahn_ctx *ctx = ahn_ctx_new(k, n, -1, &oo);
        if (!ctx) {
            printf("skip k=%" PRIu64 " n=%" PRIu64 " (no ctx)\n", k, n);
            continue;
        }
        mpz_set_ui(N, k);
        mpz_mul_2exp(N, N, n);
        mpz_sub_ui(N, N, 1);
        size_t nl = ahn_limbs(ctx);
        uint64_t *l = calloc(nl, sizeof(uint64_t));
        // random a, b in [0, N)
        for (size_t i = 0; i < nl; i++) l[i] = rng();
        to_mpz(a, l, nl);
        mpz_mod(a, a, N);
        for (size_t i = 0; i < nl; i++) l[i] = rng();
        to_mpz(b, l, nl);
        mpz_mod(b, b, N);

        ahn_num *x = ahn_num_new(ctx), *y = ahn_num_new(ctx);
        from_mpz(l, nl, a);
        ahn_set_limbs(ctx, x, l);
        fails += check(ctx, x, a, "roundtrip", k, n);

        // 20 squarings
        ahn_square_n(ctx, x, 20);
        mpz_set_ui(t, 1);
        mpz_mul_2exp(t, t, 20);
        mpz_powm(t, a, t, N);
        fails += check(ctx, x, t, "square", k, n);

        // multiply by b
        from_mpz(l, nl, b);
        ahn_set_limbs(ctx, y, l);
        ahn_mul(ctx, x, y);
        mpz_mul(t, t, b);
        mpz_mod(t, t, N);
        fails += check(ctx, x, t, "mul", k, n);

        ahn_mul_small(ctx, x, 3);
        mpz_mul_ui(t, t, 3);
        mpz_mod(t, t, N);
        fails += check(ctx, x, t, "mul_small", k, n);

        // res64 of 3^(2^m)
        ahn_set_u64(ctx, x, 3);
        ahn_square_n(ctx, x, 30);
        mpz_set_ui(t, 1);
        mpz_mul_2exp(t, t, 30);
        mpz_set_ui(a, 3);
        mpz_powm(t, a, t, N);
        fails += check(ctx, x, t, "small-start", k, n);

        printf("k=%-5" PRIu64 " n=%-6" PRIu64 " L=%-6u bits/w=%5.2f maxerr=%.4f\n", k, n,
               ahn_fft_len(ctx), (double)n / ahn_fft_len(ctx), ahn_max_roundoff(ctx));
        ahn_num_free(ctx, x);
        ahn_num_free(ctx, y);
        free(l);
        ahn_ctx_free(ctx);
    }
    mpz_clears(N, a, b, t, NULL);
    printf("%s: %d failures\n", fails ? "FAILED" : "PASSED", fails);
    return fails != 0;
}
