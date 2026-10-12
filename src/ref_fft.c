// Portable scalar reference backend. Slow but simple: a full-length complex
// mixed-radix FFT applied to real weighted input. Used as a correctness oracle
// for the assembly backends.
#include "internal.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef double complex cplx;

typedef struct {
    cplx *tw;   // tw[j] = exp(-2*pi*i*j/L)
    cplx *a, *b;
} ref_state;

static uint32_t smallest_factor(uint32_t n) {
    for (uint32_t p = 2; p * p <= n; p++) {
        if (n % p == 0) return p;
    }
    return n;
}

// Out-of-place recursive decimation-in-time FFT (forward, e^{-i...}).
static void fft_rec(const ref_state *st, uint32_t L, const cplx *in, size_t istride,
                    cplx *out, uint32_t n) {
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    uint32_t p = smallest_factor(n), m = n / p;
    for (uint32_t r = 0; r < p; r++) fft_rec(st, L, in + r * istride, istride * p, out + r * m, m);
    cplx t[8], u[8];
    cplx *tp = p <= 8 ? t : malloc(2 * p * sizeof(cplx));
    cplx *up = p <= 8 ? u : tp + p;
    uint32_t ln = L / n, lp = L / p;
    for (uint32_t k = 0; k < m; k++) {
        for (uint32_t r = 0; r < p; r++) tp[r] = out[r * m + k] * st->tw[((uint64_t)r * k * ln) % L];
        for (uint32_t q = 0; q < p; q++) {
            cplx s = 0;
            for (uint32_t r = 0; r < p; r++) s += tp[r] * st->tw[((r * q) % p) * lp];
            up[q] = s;
        }
        for (uint32_t q = 0; q < p; q++) out[q * m + k] = up[q];
    }
    if (p > 8) free(tp);
}

static int ref_init(kite_ctx *ctx) {
    ref_state *st = calloc(1, sizeof *st);
    if (!st) return -1;
    uint32_t L = ctx->L;
    st->tw = kite_alloc(L * sizeof(cplx));
    st->a = kite_alloc(L * sizeof(cplx));
    st->b = kite_alloc(L * sizeof(cplx));
    ctx->bk = st;
    if (!st->tw || !st->a || !st->b) return -1;
    for (uint32_t j = 0; j < L; j++) {
        double x = 2.0 * j / L;
        st->tw[j] = __cospi(x) - I * __sinpi(x);
    }
    ctx->num_doubles = L;
    return 0;
}

static void ref_free(kite_ctx *ctx) {
    ref_state *st = ctx->bk;
    kite_free_aligned(st->tw);
    kite_free_aligned(st->a);
    kite_free_aligned(st->b);
    free(st);
    ctx->bk = NULL;
}

static void ref_to_digits(kite_ctx *ctx, const kite_num *x, int64_t *dig) {
    for (uint32_t j = 0; j < ctx->L; j++) dig[j] = (int64_t)x->d[j];
}

static void ref_from_digits(kite_ctx *ctx, kite_num *x, const int64_t *dig) {
    for (uint32_t j = 0; j < ctx->L; j++) x->d[j] = (double)dig[j];
}

static void forward(kite_ctx *ctx, const kite_num *x, cplx *dst) {
    ref_state *st = ctx->bk;
    uint32_t L = ctx->L;
    cplx *in = st->b == dst ? st->a : st->b;
    for (uint32_t j = 0; j < L; j++) in[j] = x->d[j] * ctx->w[j];
    fft_rec(st, L, in, 1, dst, L);
}

// dst holds the pointwise product in the frequency domain; finish the
// inverse transform, unweight, round, and carry into x.
static void inverse_carry(kite_ctx *ctx, cplx *prod, kite_num *x) {
    ref_state *st = ctx->bk;
    uint32_t L = ctx->L;
    cplx *out = prod == st->a ? st->b : st->a;
    for (uint32_t j = 0; j < L; j++) prod[j] = conj(prod[j]);
    fft_rec(st, L, prod, 1, out, L);
    int64_t *dig = ctx->tmpdig;
    double maxerr = ctx->maxerr, scale = 1.0 / L;
    for (uint32_t j = 0; j < L; j++) {
        double t = creal(out[j]) * scale * ctx->oscale[j];
        double r = rint(t);
        double e = fabs(t - r);
        if (e > maxerr) maxerr = e;
        dig[j] = (int64_t)r;
    }
    ctx->maxerr = maxerr;
    kite_carry_digits(ctx, dig);
    ref_from_digits(ctx, x, dig);
}

static void ref_square(kite_ctx *ctx, kite_num *x) {
    ref_state *st = ctx->bk;
    forward(ctx, x, st->a);
    for (uint32_t j = 0; j < ctx->L; j++) st->a[j] *= st->a[j];
    inverse_carry(ctx, st->a, x);
}

static void ref_mul(kite_ctx *ctx, kite_num *x, const kite_num *y) {
    ref_state *st = ctx->bk;
    uint32_t L = ctx->L;
    cplx *fy = malloc(L * sizeof(cplx));
    forward(ctx, y, st->a);
    memcpy(fy, st->a, L * sizeof(cplx));
    forward(ctx, x, st->a);
    for (uint32_t j = 0; j < L; j++) st->a[j] *= fy[j];
    free(fy);
    inverse_carry(ctx, st->a, x);
}

static void ref_mul_small(kite_ctx *ctx, kite_num *x, int64_t m) {
    int64_t *dig = ctx->tmpdig;
    ref_to_digits(ctx, x, dig);
    for (uint32_t j = 0; j < ctx->L; j++) dig[j] *= m;
    kite_carry_digits(ctx, dig);
    ref_from_digits(ctx, x, dig);
}

const kite_backend_ops kite_ref_ops = {
    .name = "ref",
    .init = ref_init,
    .free = ref_free,
    .to_digits = ref_to_digits,
    .from_digits = ref_from_digits,
    .square = ref_square,
    .mul = ref_mul,
    .mul_small = ref_mul_small,
};
