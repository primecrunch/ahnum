// Context setup, digit layout, weights, conversion to/from full integers.
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void *ahn_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 128, bytes ? bytes : 128)) return NULL;
    memset(p, 0, bytes);
    return p;
}

void ahn_free_aligned(void *p) { free(p); }

bool ahn_fft_len_ok(uint32_t L) {
    if (L < 16 || (L & 1)) return false;
    while (L % 2 == 0) L /= 2;
    while (L % 3 == 0) L /= 3;
    while (L % 5 == 0) L /= 5;
    while (L % 7 == 0) L /= 7;
    return L == 1;
}

// Safe average bits per word. Fitted from measured round-off (tools/fit_bpw.py:
// max error <= 0.25 over a few hundred squarings) on the NEON backend:
//   21.44 - 0.305*(log2 L - 12) - 0.50*log2 k
// less a margin so that the max error over a full test stays below 0.4.
#define BPW_MARGIN 0.2
double ahn_safe_bpw(uint32_t L, uint64_t k) {
    return 21.44 - 0.305 * (log2((double)L) - 12) - 0.50 * log2((double)k) - BPW_MARGIN;
}

// Measured squaring times (us) per FFT length, loaded from a tuning file with
// lines "L us [order]" (see `ahbench tune`; order as ahn_opts.plan_order bits,
// 0 if absent). Path: $AHN_FFT_TIMES, else ~/.ahnum/fft_times.txt.
static struct {
    int loaded;
    uint32_t n;
    uint32_t *L;
    double *us;
    uint8_t *ord;
} g_times;

static void load_times(void) {
    if (g_times.loaded) return;
    g_times.loaded = 1;
    char path[1024];
    const char *p = getenv("AHN_FFT_TIMES");
    if (!p) {
        const char *home = getenv("HOME");
        if (!home) return;
        snprintf(path, sizeof path, "%s/.ahnum/fft_times.txt", home);
        p = path;
    }
    FILE *f = fopen(p, "r");
    if (!f) return;
    uint32_t cap = 0, L, ord;
    double us;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        int got = sscanf(line, "%u %lf %u", &L, &us, &ord);
        if (got < 2) continue;
        if (got < 3 || ord > 3) ord = 0;
        if (g_times.n == cap) {
            cap = cap ? 2 * cap : 256;
            g_times.L = realloc(g_times.L, cap * sizeof(uint32_t));
            g_times.us = realloc(g_times.us, cap * sizeof(double));
            g_times.ord = realloc(g_times.ord, cap);
        }
        g_times.L[g_times.n] = L;
        g_times.ord[g_times.n] = (uint8_t)ord;
        g_times.us[g_times.n++] = us;
    }
    fclose(f);
}

uint32_t ahn_tuned_plan_order(uint32_t L) {
    load_times();
    for (uint32_t i = 0; i < g_times.n; i++) {
        if (g_times.L[i] == L) return g_times.ord[i];
    }
    return 0;
}

static double measured_us(uint32_t L) {
    for (uint32_t i = 0; i < g_times.n; i++) {
        if (g_times.L[i] == L) return g_times.us[i];
    }
    return -1;
}

// Heuristic relative cost when no measurement is available: odd radices cost
// more per point than radix 8/4/2.
static double cost_model(uint32_t L) {
    double c = log2((double)L);
    uint32_t m = L;
    while (m % 3 == 0) { m /= 3; c += 0.6; }
    while (m % 5 == 0) { m /= 5; c += 1.0; }
    while (m % 7 == 0) { m /= 7; c += 1.4; }
    return (double)L * c;
}

static bool len_usable(uint32_t L) { return L % 64 == 0 && ahn_fft_len_ok(L); }

uint32_t ahn_choose_fft_len(uint64_t k, uint64_t n) {
    if (n < 2048) {
        // tiny numbers (reference backend): short power-of-two lengths
        for (uint32_t L = 16; L <= n; L *= 2) {
            if ((double)n / L <= ahn_safe_bpw(L, k)) return L;
        }
        return 0;
    }
    uint32_t Lmin = 0;
    for (uint32_t L = 64; L && L < (1u << 28); L += 64) {
        if (len_usable(L) && (double)n / L <= ahn_safe_bpw(L, k)) {
            Lmin = L;
            break;
        }
    }
    if (!Lmin) return 0;
    load_times();
    // among safe lengths up to 30% longer, pick the cheapest
    uint32_t best = Lmin;
    double bestc = 1e300;
    int have_all = 1;
    for (uint32_t L = Lmin; L <= Lmin + Lmin / 10 * 3; L += 64) {
        if (!len_usable(L)) continue;
        if (measured_us(L) < 0) have_all = 0;
    }
    for (uint32_t L = Lmin; L <= Lmin + Lmin / 10 * 3; L += 64) {
        if (!len_usable(L)) continue;
        double c = have_all ? measured_us(L) : cost_model(L);
        if (c < bestc) {
            bestc = c;
            best = L;
        }
    }
    return best;
}

uint32_t ahn_next_fft_len(uint32_t L) {
    for (uint32_t M = (L / 64 + 1) * 64; M && M < (1u << 28); M += 64) {
        if (ahn_fft_len_ok(M)) return M;
    }
    return 0;
}

// log2(k) - floor(log2(k)) as a 64-bit binary fraction, by repeated squaring in
// fixed point (each square yields one bit). Accurate to a few units of 2^-60,
// far below what double weights can resolve.
static uint64_t log2_frac64(uint64_t k) {
    int e = 63 - __builtin_clzll(k);
    // y in [1, 2) with 62 fractional bits
    unsigned __int128 y = (unsigned __int128)k << (62 - e);
    uint64_t f = 0;
    for (int i = 0; i < 64; i++) {
        y = (y * y) >> 62;
        f <<= 1;
        if (y >> 63) {  // y >= 2
            f |= 1;
            y >>= 1;
        }
    }
    return f;
}

// T' = n + floor(log2 k) + frac/2^64. Everything (digit boundaries and weights)
// is defined from T' with exact integer arithmetic; z^L = 2^T' differs from
// k*2^n by a relative 2^-64, which is below double precision.
uint64_t ahn_s_at(const ahn_ctx *ctx, uint64_t j) {
    // s_j = ceil(j*T'/L), T'*2^64 = (n + ilog2k)*2^64 + lam
    unsigned __int128 num = (unsigned __int128)j * ctx->lam;
    unsigned __int128 den = (unsigned __int128)ctx->L << 64;
    uint64_t ik = (uint64_t)(63 - __builtin_clzll(ctx->k));
    unsigned __int128 whole = (unsigned __int128)j * (ctx->n + ik);
    // ceil((whole*2^64 + num) / (L*2^64)) without overflowing 128 bits:
    // whole = qa*L + ra; then ceil((ra*2^64 + num)/(L*2^64)) + qa
    unsigned __int128 qa = whole / ctx->L, ra = whole % ctx->L;
    unsigned __int128 top = (ra << 64) + num;
    return (uint64_t)(qa + (top + den - 1) / den);
}

// w_j = 2^{s_j - j*T'/L}; the exponent is an exact ratio of integers.
double ahn_weight(const ahn_ctx *ctx, uint64_t j, uint64_t sj) {
    uint64_t ik = (uint64_t)(63 - __builtin_clzll(ctx->k));
    unsigned __int128 whole = (unsigned __int128)j * (ctx->n + ik);
    unsigned __int128 qa = whole / ctx->L, ra = whole % ctx->L;
    // s_j - j*T'/L = (s_j - qa) - (ra*2^64 + j*lam) / (L*2^64)
    unsigned __int128 num = ((unsigned __int128)(sj - qa) * ctx->L << 64) -
                            ((ra << 64) + (unsigned __int128)j * ctx->lam);
    // num / (L*2^64) in [0, 1): take the top 64 bits of the fraction
    double frac = (double)(num / ctx->L) / 18446744073709551616.0;
    return exp2(frac);
}

static int build_layout(ahn_ctx *ctx) {
    uint32_t L = ctx->L;
    uint64_t n = ctx->n, k = ctx->k;
    ctx->lam = log2_frac64(k);
    uint64_t s = 0;
    ctx->top_t = L;
    for (uint32_t j = 0; j < L; j++) {
        uint64_t s_next = ahn_s_at(ctx, j + 1);
        if (s_next < s || s_next - s > 40) return -1;
        ctx->bits[j] = (uint8_t)(s_next - s);
        if (s_next > n && ctx->top_t == L) {
            ctx->top_t = j;
            ctx->top_st = s;
        }
        ctx->w[j] = ahn_weight(ctx, j, s);
        ctx->oscale[j] = (double)k / ctx->w[j];
        s = s_next;
    }
    ctx->sL = s;
    if (ctx->top_t == L) ctx->top_st = s;  // k == 1: only the final carry folds
    ctx->top_m = (uint32_t)(n - ctx->top_st);
    if (ctx->sL - ctx->top_st > 100 || ctx->top_t == 0) return -1;
    return 0;
}

ahn_ctx *ahn_ctx_new(uint64_t k, uint64_t n, int32_t c, const ahn_opts *o) {
    if (c != -1 || k == 0 || (k & 1) == 0 || n < 8) return NULL;
    if (k >= (1ull << 30)) return NULL;
    ahn_ctx *ctx = calloc(1, sizeof *ctx);
    if (!ctx) return NULL;
    ctx->k = k;
    ctx->n = n;
    ctx->c = c;
    ctx->L = o && o->fft_len ? o->fft_len : ahn_choose_fft_len(k, n);
    ctx->threads = o && o->threads ? o->threads : 1;
    ctx->plan_order = o && o->plan_order ? (o->plan_order - 1) & 3 : ahn_tuned_plan_order(ctx->L);
    ctx->err_every = 1;
    if (!ahn_fft_len_ok(ctx->L) || ctx->L > n) goto fail;

    ctx->bits = malloc(ctx->L);
    ctx->w = ahn_alloc(ctx->L * sizeof(double));
    ctx->oscale = ahn_alloc(ctx->L * sizeof(double));
    ctx->tmpdig = ahn_alloc(ctx->L * sizeof(int64_t));
    ctx->nlimbs = (n + 64 + 63) / 64;
    ctx->Nlimbs = calloc(ctx->nlimbs, sizeof(uint64_t));
    if (!ctx->bits || !ctx->w || !ctx->oscale || !ctx->tmpdig || !ctx->Nlimbs) goto fail;
    if (build_layout(ctx)) goto fail;

    // N = k*2^n - 1
    {
        uint64_t sh = n % 64, lw = n / 64;
        ctx->Nlimbs[lw] = k << sh;
        if (sh && lw + 1 < ctx->nlimbs) ctx->Nlimbs[lw + 1] = k >> (64 - sh);
        for (size_t i = 0; i < ctx->nlimbs; i++) {  // subtract 1
            if (ctx->Nlimbs[i]--) break;
        }
    }

    ahn_backend b = o ? o->backend : AHN_BACKEND_AUTO;
    switch (b) {
    case AHN_BACKEND_REF:
        ctx->ops = &ahn_ref_ops;
        break;
    case AHN_BACKEND_AUTO:
    case AHN_BACKEND_NEON:
        ctx->ops = &ahn_neon_ops;
        break;
    case AHN_BACKEND_SME:
        ctx->ops = &ahn_sme_ops;
        break;
    case AHN_BACKEND_C2P:
        ctx->ops = &ahn_f2_ops;
        break;
    default:
        goto fail;
    }
    if (ctx->ops->init(ctx)) {
        // tiny sizes the fast backend cannot shape fall back to the reference
        if (b != AHN_BACKEND_AUTO || ctx->ops == &ahn_ref_ops) goto fail;
        if (ctx->bk) ctx->ops->free(ctx);
        ctx->ops = &ahn_ref_ops;
        if (ctx->ops->init(ctx)) goto fail;
    }
    return ctx;
fail:
    ahn_ctx_free(ctx);
    return NULL;
}

void ahn_ctx_free(ahn_ctx *ctx) {
    if (!ctx) return;
    if (ctx->ops && ctx->bk) ctx->ops->free(ctx);
    free(ctx->bits);
    ahn_free_aligned(ctx->w);
    ahn_free_aligned(ctx->oscale);
    ahn_free_aligned(ctx->tmpdig);
    free(ctx->Nlimbs);
    free(ctx);
}

uint32_t ahn_fft_len(const ahn_ctx *ctx) { return ctx->L; }
const char *ahn_backend_name(const ahn_ctx *ctx) { return ctx->ops->name; }
double ahn_max_roundoff(const ahn_ctx *ctx) { return ctx->maxerr; }
void ahn_reset_roundoff(ahn_ctx *ctx) { ctx->maxerr = 0; }
void ahn_set_roundoff_check(ahn_ctx *ctx, uint32_t every) { ctx->err_every = every ? every : 1; }
size_t ahn_limbs(const ahn_ctx *ctx) { return ctx->nlimbs; }

ahn_num *ahn_num_new(ahn_ctx *ctx) {
    ahn_num *x = malloc(sizeof *x);
    if (!x) return NULL;
    x->d = ahn_alloc(ctx->num_doubles * sizeof(double));
    if (!x->d) {
        free(x);
        return NULL;
    }
    return x;
}

void ahn_num_free(ahn_ctx *ctx, ahn_num *x) {
    (void)ctx;
    if (!x) return;
    ahn_free_aligned(x->d);
    free(x);
}

void ahn_copy(ahn_ctx *ctx, ahn_num *dst, const ahn_num *src) {
    memcpy(dst->d, src->d, ctx->num_doubles * sizeof(double));
}

void ahn_square(ahn_ctx *ctx, ahn_num *x) { ctx->ops->square(ctx, x); }

void ahn_square_n(ahn_ctx *ctx, ahn_num *x, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) ctx->ops->square(ctx, x);
}

void ahn_mul(ahn_ctx *ctx, ahn_num *x, const ahn_num *y) { ctx->ops->mul(ctx, x, y); }
void ahn_mul_small(ahn_ctx *ctx, ahn_num *x, int64_t m) { ctx->ops->mul_small(ctx, x, m); }

// Fold the top region (digits t..L-1 plus carry c at 2^{s_L}) so that it
// holds a value in [-k*2^m/2, k*2^m/2) (m = n - s_t); returns the amount to
// add into digit 0.
static int64_t fold_top(const ahn_ctx *ctx, int64_t *dig, int64_t c) {
    uint32_t L = ctx->L, t = ctx->top_t;
    __int128 H = (__int128)c << (ctx->sL - ctx->top_st);
    uint64_t pos = 0;
    for (uint32_t j = t; j < L; j++) {
        H += (__int128)dig[j] << pos;
        pos += ctx->bits[j];
    }
    unsigned m = ctx->top_m;
    __int128 lo = H & (((__int128)1 << m) - 1);
    __int128 hi = H >> m;  // floor
    __int128 kk = (__int128)ctx->k;
    __int128 q = hi >= 0 ? hi / kk : -((-hi + kk - 1) / kk);
    __int128 top = lo + ((hi - q * kk) << m);
    __int128 span = kk << m;
    if (2 * top >= span) {  // subtract k*2^n == 1
        top -= span;
        q += 1;
    }
    for (uint32_t j = t; j < L - 1; j++) {
        unsigned b = ctx->bits[j];
        __int128 qq = (top + ((__int128)1 << (b - 1))) >> b;
        dig[j] = (int64_t)(top - (qq << b));
        top = qq;
    }
    if (t < L) dig[L - 1] = (int64_t)top;  // for k == 1 there is no top region
    return (int64_t)q;
}

// Balanced-digit carry propagation with the top fold.
void ahn_carry_digits(const ahn_ctx *ctx, int64_t *dig) {
    uint32_t L = ctx->L;
    int64_t carry = 0;
    for (uint32_t j = 0; j < L; j++) {
        unsigned b = ctx->bits[j];
        int64_t v = dig[j] + carry;
        int64_t q = (v + (1ll << (b - 1))) >> b;  // digit in [-2^(b-1), 2^(b-1))
        dig[j] = v - (q << b);
        carry = q;
    }
    for (int pass = 0; pass < 64; pass++) {
        // carry is at place 2^{s_L}; the fold returns an amount for digit 0
        carry = fold_top(ctx, dig, carry);
        uint32_t j;
        for (j = 0; j < ctx->top_t && carry; j++) {
            unsigned b = ctx->bits[j];
            int64_t v = dig[j] + carry;
            int64_t q = (v + (1ll << (b - 1))) >> b;
            dig[j] = v - (q << b);
            carry = q;
        }
        if (!carry) return;
        // the ripple reached the top region: add it there and fold again
        // (for k == 1 there is no top region and the carry is at 2^{s_L})
        if (ctx->top_t < ctx->L) {
            dig[ctx->top_t] += carry;
            carry = 0;
        }
    }
}

// ---- conversion between digits and full integers -------------------------

static void limbs_add_bits(uint64_t *a, size_t na, uint64_t pos, uint64_t v) {
    // a += v << pos, v < 2^64
    size_t w = pos / 64;
    unsigned sh = pos % 64;
    unsigned __int128 t = (unsigned __int128)v << sh;
    uint64_t lo = (uint64_t)t, hi = (uint64_t)(t >> 64);
    unsigned __int128 s = (unsigned __int128)a[w] + lo;
    a[w] = (uint64_t)s;
    uint64_t c = (uint64_t)(s >> 64) + hi;
    for (size_t i = w + 1; c && i < na; i++) {
        s = (unsigned __int128)a[i] + c;
        a[i] = (uint64_t)s;
        c = (uint64_t)(s >> 64);
    }
}

static int limbs_cmp(const uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t i = n; i-- > 0;) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

static void limbs_sub(uint64_t *a, const uint64_t *b, size_t n) {
    uint64_t borrow = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned __int128 d = (unsigned __int128)a[i] - b[i] - borrow;
        a[i] = (uint64_t)d;
        borrow = (uint64_t)(d >> 64) & 1;
    }
}

// Bits [pos, pos+cnt) of a, cnt <= 64.
static uint64_t limbs_get_bits(const uint64_t *a, size_t na, uint64_t pos, unsigned cnt) {
    size_t w = pos / 64;
    unsigned sh = pos % 64;
    if (w >= na) return 0;
    unsigned __int128 v = a[w] >> sh;
    if (sh && w + 1 < na) v |= (unsigned __int128)a[w + 1] << (64 - sh);
    if (!sh && w + 1 < na) v |= (unsigned __int128)a[w + 1] << 64;
    return cnt >= 64 ? (uint64_t)v : (uint64_t)v & ((1ull << cnt) - 1);
}

// a = (a mod 2^n) + q + r*2^n where a>>n = q*k + r; then reduce below N.
// Requires a>>n < 2^120; uses k*2^n == 1 (mod N).
static void reduce_mod_N(const ahn_ctx *ctx, uint64_t *a, size_t na) {
    uint64_t n = ctx->n, k = ctx->k;
    for (int pass = 0; pass < 4; pass++) {
        size_t w = n / 64;
        unsigned sh = n % 64;
        unsigned __int128 h = 0;
        for (size_t i = na; i-- > w;) h = (h << 64) | a[i];
        h >>= sh;
        if (!h) break;
        if (sh) a[w] &= (1ull << sh) - 1;
        else a[w] = 0;
        for (size_t i = w + 1; i < na; i++) a[i] = 0;
        unsigned __int128 q = h / k;
        uint64_t r = (uint64_t)(h % k);
        limbs_add_bits(a, na, n, r);
        limbs_add_bits(a, na, 0, (uint64_t)q);
        if (q >> 64) limbs_add_bits(a, na, 64, (uint64_t)(q >> 64));
    }
    while (limbs_cmp(a, ctx->Nlimbs, ctx->nlimbs) >= 0) limbs_sub(a, ctx->Nlimbs, ctx->nlimbs);
}

// k * (sum of the positive or negative digits) mod N, into a (na limbs).
static void signed_part_mod_N(const ahn_ctx *ctx, const int64_t *dig, int sign, uint64_t *a,
                              size_t na) {
    memset(a, 0, na * sizeof(uint64_t));
    uint64_t s = 0;
    for (uint32_t j = 0; j < ctx->L; j++) {
        int64_t d = dig[j];
        if (sign > 0 && d > 0) limbs_add_bits(a, na, s, (uint64_t)d);
        if (sign < 0 && d < 0) limbs_add_bits(a, na, s, (uint64_t)-d);
        s += ctx->bits[j];
    }
    unsigned __int128 c = 0;
    for (size_t i = 0; i < na; i++) {
        c += (unsigned __int128)a[i] * ctx->k;
        a[i] = (uint64_t)c;
        c >>= 64;
    }
    reduce_mod_N(ctx, a, na);
}

void ahn_get_limbs(ahn_ctx *ctx, const ahn_num *x, uint64_t *limbs) {
    int64_t *dig = ctx->tmpdig;
    ctx->ops->to_digits(ctx, x, dig);
    size_t na = (ctx->sL + 64) / 64 + 2;
    uint64_t *p = calloc(na, sizeof(uint64_t)), *m = calloc(na, sizeof(uint64_t));
    // stored V = x/k, so x = k*V
    signed_part_mod_N(ctx, dig, 1, p, na);
    signed_part_mod_N(ctx, dig, -1, m, na);
    if (limbs_cmp(p, m, ctx->nlimbs) < 0) {
        uint64_t carry = 0;  // p += N
        for (size_t i = 0; i < ctx->nlimbs; i++) {
            unsigned __int128 t = (unsigned __int128)p[i] + ctx->Nlimbs[i] + carry;
            p[i] = (uint64_t)t;
            carry = (uint64_t)(t >> 64);
        }
    }
    limbs_sub(p, m, ctx->nlimbs);
    memcpy(limbs, p, ctx->nlimbs * sizeof(uint64_t));
    free(p);
    free(m);
}

void ahn_set_limbs(ahn_ctx *ctx, ahn_num *x, const uint64_t *limbs) {
    size_t nl = ctx->nlimbs + 1;
    uint64_t *a = calloc(nl, sizeof(uint64_t));
    memcpy(a, limbs, ctx->nlimbs * sizeof(uint64_t));
    // V = x*2^n mod N: x = q*k + r  ->  V = q + r*2^n
    unsigned __int128 r = 0;
    for (size_t i = nl; i-- > 0;) {
        r = (r << 64) | a[i];
        a[i] = (uint64_t)(r / ctx->k);
        r %= ctx->k;
    }
    limbs_add_bits(a, nl, ctx->n, (uint64_t)r);
    if (limbs_cmp(a, ctx->Nlimbs, ctx->nlimbs) >= 0 && !a[nl - 1]) {
        limbs_sub(a, ctx->Nlimbs, ctx->nlimbs);
    }
    int64_t *dig = ctx->tmpdig;
    uint64_t s = 0;
    for (uint32_t j = 0; j + 1 < ctx->L; j++) {
        dig[j] = (int64_t)limbs_get_bits(a, nl, s, ctx->bits[j]);
        s += ctx->bits[j];
    }
    dig[ctx->L - 1] = (int64_t)limbs_get_bits(a, nl, s, 64);  // < 2^{s_L - s_{L-1}}
    free(a);
    ahn_carry_digits(ctx, dig);
    ctx->ops->from_digits(ctx, x, dig);
}

void ahn_set_u64(ahn_ctx *ctx, ahn_num *x, uint64_t v) {
    uint64_t *t = calloc(ctx->nlimbs, sizeof(uint64_t));
    t[0] = v;
    ahn_set_limbs(ctx, x, t);
    free(t);
}

uint64_t ahn_res64(ahn_ctx *ctx, const ahn_num *x) {
    uint64_t *t = malloc(ctx->nlimbs * sizeof(uint64_t));
    ahn_get_limbs(ctx, x, t);
    uint64_t r = t[0];
    free(t);
    return r;
}
