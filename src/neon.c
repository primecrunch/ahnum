// NEON backend: the two-pass structure of fast2p.c driven by the generated
// assembly kernels.
//
// Memory layout of a number: H = R*C complex elements, row-major, row stride
// `rowstride` bytes (C*16 plus padding against cache-set conflicts). Within a
// row, columns are stored in 8-column blocks of 128 bytes [re c0..c7][im c0..c7]
// ("layout b"); pass 1 works on one block per row (a group of 8 columns).
// Pass 2 transposes a row pair (p, p') into a scratch buffer whose element q is
// [re_p(q), re_p'(q), im_p(q), im_p'(q)], so every row-FFT butterfly is
// lane-uniform and the real-split partner of lane 0 at q is lane 1 at C-1-q.
//
// Weights are compressed: for digit j = J + i (J = first digit of a group row
// segment, i < 16), w_j = A[g,r] * B[i] * 2^{-e_j}, with e_j and the "big digit"
// flag b_j (bits = b + b_j) packed in a per-row mask (e: bit i, big: bit 16+i).
#include "internal.h"
#include "kernels.h"
#include "threads.h"

#include <math.h>
#include <sys/sysctl.h>
#include <stdlib.h>
#include <string.h>

#define G 8

typedef void (*stage_fn)(double *, size_t, size_t, size_t, size_t, size_t, const double *);
typedef void (*t4_fn)(double *, size_t, size_t, size_t, size_t, ptrdiff_t);

typedef struct {
    uint8_t p;
    uint32_t s, nblocks;
    const double *tw;  // s*(p-1) packed complex
    // [0]: layout a (pass-2 pair scratch), [1]: layout b (8-column blocks)
    stage_fn tw_f[2], notw_f[2];
    t4_fn dif_t4[2], dit_t4[2];
    stage_fn dit_tw[2], dit_notw[2];
    // radix-8 only: complex 8x8 matrices per j for the SME kernel (1 KB each),
    // DIF: diag(w_j) F8, DIT: conj(F8) diag(conj w_j)
    double *mdif, *mdit;
} stage;

typedef struct {
    uint32_t n, nst;
    stage st[32];
    double *twmem;
    uint32_t *rev, *pos;
} nplan;

typedef struct {
    uint32_t H, R, C, ngroups;
    size_t rowstride;  // bytes
    nplan pr, pc;
    double *t4;        // four-step twiddles, same layout as data
    int sme;           // radix-8 column stages on the SME unit
    double *A, *IA;    // [g*R + r]
    uint64_t *mask;    // [g*R + r]
    double B[16], IB[16], IBs[16];
    double *nconsts;   // asm normaliser constants (see gen_norm)
    double *ntab;      // table-driven normaliser records (NULL = compressed)
    double base_small, ibase_small;  // 2^b, 2^-b
    double scale;                    // k / (4H)
    uint32_t bsmall;
    // pass 2
    uint32_t *partner;
    double *rs;     // per row position p: packed [re, im] of w_H^{k_r}
    double *cs;     // per column position q: packed [re, im] of w_C^{k_c}
    double *yscr;   // mul operand: H elements in scratch format, by row pair
    uint32_t (*pairs)[2];  // row pairs in processing order; pairs[0] is (k_r=0, R/2)
    uint32_t npairs;
    // threading
    ahn_pool *pool;
    int nth;
    uint32_t gstart[65];   // thread t owns groups [gstart[t], gstart[t+1])
    double *scr[64];       // per-thread pass-2 scratch (C elements)
    double *carry[64];     // per-thread carries (R)
    double maxerr_t[64];
    double wrapq;
    // current job
    ahn_ctx *ctx;
    double *d;
    int mul;
} neon_state;

// ---- plans --------------------------------------------------------------

static int factorize(uint32_t n, uint8_t *radix) {
    int m = 0;
    static const uint8_t pr[] = {8, 4, 2, 7, 5, 3};
    for (int i = 0; i < 6; i++) {
        while (n % pr[i] == 0) {
            radix[m++] = pr[i];
            n /= pr[i];
        }
    }
    return n == 1 ? m : -1;
}

static void set_fns(stage *s) {
#define CASE(P)                          \
    case P:                              \
        s->tw_f[0] = ahn_dif_r##P;                \
        s->notw_f[0] = ahn_dif_r##P##_notw;       \
        s->dif_t4[0] = ahn_dif_r##P##_t4;         \
        s->dit_t4[0] = ahn_dit_r##P##_t4;         \
        s->dit_tw[0] = ahn_dit_r##P;              \
        s->dit_notw[0] = ahn_dit_r##P##_notw;     \
        s->tw_f[1] = ahn_dif_r##P##_b;            \
        s->notw_f[1] = ahn_dif_r##P##_notw_b;     \
        s->dif_t4[1] = ahn_dif_r##P##_t4_b;       \
        s->dit_t4[1] = ahn_dit_r##P##_t4_b;       \
        s->dit_tw[1] = ahn_dit_r##P##_b;          \
        s->dit_notw[1] = ahn_dit_r##P##_notw_b;   \
        break;
    switch (s->p) {
        CASE(2)
        CASE(3)
        CASE(4)
        CASE(5)
        CASE(7)
        CASE(8)
    }
#undef CASE
}

// first4: start with a radix-4 stage (the row FFT fuses it with the transpose)
static int nplan_init(nplan *pl, uint32_t n, int first4) {
    uint8_t radix[32];
    int m;
    if (first4) {
        if (n % 4) return -1;
        m = factorize(n / 4, radix + 1);
        if (m < 0) return -1;
        radix[0] = 4;
        m += 1;
    } else {
        m = factorize(n, radix);
    }
    if (m < 1) return -1;
    pl->n = n;
    pl->nst = (uint32_t)m;
    size_t tot = 0;
    uint32_t sz = n;
    for (int i = 0; i < m; i++) {
        uint32_t s = sz / radix[i];
        tot += (size_t)s * (radix[i] - 1) * 2;
        sz = s;
    }
    pl->twmem = ahn_alloc((tot + 2) * sizeof(double));
    pl->rev = malloc(n * sizeof(uint32_t));
    pl->pos = malloc(n * sizeof(uint32_t));
    double *tp = pl->twmem;
    sz = n;
    for (int i = 0; i < m; i++) {
        stage *st = &pl->st[i];
        st->p = radix[i];
        st->s = sz / st->p;
        st->nblocks = n / sz;
        st->tw = tp;
        set_fns(st);
        for (uint32_t j = 0; j < st->s; j++) {
            for (uint32_t t = 1; t < st->p; t++) {
                double a = 2.0 * (double)((uint64_t)t * j % sz) / sz;
                *tp++ = __cospi(a);
                *tp++ = -__sinpi(a);
            }
        }
        if (st->p == 8) {
            st->mdif = ahn_alloc((size_t)st->s * 128 * sizeof(double));
            st->mdit = ahn_alloc((size_t)st->s * 128 * sizeof(double));
            for (uint32_t j = 0; j < st->s; j++) {
                for (uint32_t u = 0; u < 8; u++) {
                    for (uint32_t t = 0; t < 8; t++) {
                        // f = w8^{tu}, w = w_sz^{t j} (DIF) or w_sz^{u j} (DIT)
                        double af = 2.0 * ((t * u) % 8) / 8;
                        double fr = __cospi(af), fi = -__sinpi(af);
                        double aw = 2.0 * (double)((uint64_t)t * j % sz) / sz;
                        double wr = __cospi(aw), wi = -__sinpi(aw);
                        double *m = st->mdif + j * 128 + u * 16;
                        m[t] = fr * wr - fi * wi;
                        m[8 + t] = fr * wi + fi * wr;
                        aw = 2.0 * (double)((uint64_t)u * j % sz) / sz;
                        wr = __cospi(aw);
                        wi = -__sinpi(aw);
                        // conj(f) * conj(w)
                        m = st->mdit + j * 128 + u * 16;
                        m[t] = fr * wr - fi * wi;
                        m[8 + t] = -(fr * wi + fi * wr);
                    }
                }
            }
        }
        sz = st->s;
    }
    for (uint32_t k = 0; k < n; k++) {
        uint32_t kk = k, base = 0, size = n;
        for (int i = 0; i < m; i++) {
            uint32_t p = radix[i];
            size /= p;
            base += (kk % p) * size;
            kk /= p;
        }
        pl->pos[k] = base;
        pl->rev[base] = k;
    }
    return 0;
}

static void nplan_free(nplan *pl) {
    for (uint32_t i = 0; i < pl->nst; i++) {
        ahn_free_aligned(pl->st[i].mdif);
        ahn_free_aligned(pl->st[i].mdit);
    }
    ahn_free_aligned(pl->twmem);
    free(pl->rev);
    free(pl->pos);
}

// Forward DIF over n elements at base (element stride es, nlg lane-groups);
// the last stage uses the four-step twiddle table if delta_ok.
static void plan_fwd(const nplan *pl, int lay, double *base, size_t es, size_t nlg, int t4,
                     ptrdiff_t delta) {
    for (uint32_t i = 0; i < pl->nst; i++) {
        const stage *st = &pl->st[i];
        size_t bs = (size_t)st->s * st->p * es;
        if (st->s == 1) {
            if (t4) st->dif_t4[lay](base, es, nlg, st->nblocks, bs, delta);
            else st->notw_f[lay](base, es, nlg, st->nblocks, bs, 1, NULL);
        } else {
            st->tw_f[lay](base, es, nlg, st->nblocks, bs, st->s, st->tw);
        }
    }
}

static void plan_inv(const nplan *pl, int lay, double *base, size_t es, size_t nlg, int t4,
                     ptrdiff_t delta) {
    for (uint32_t i = pl->nst; i-- > 0;) {
        const stage *st = &pl->st[i];
        size_t bs = (size_t)st->s * st->p * es;
        if (st->s == 1) {
            if (t4) st->dit_t4[lay](base, es, nlg, st->nblocks, bs, delta);
            else st->dit_notw[lay](base, es, nlg, st->nblocks, bs, 1, NULL);
        } else {
            st->dit_tw[lay](base, es, nlg, st->nblocks, bs, st->s, st->tw);
        }
    }
}

// ---- layout -------------------------------------------------------------

// Real part of element (r, c); the imaginary part is IM doubles further on.
#define IM 8
static inline double *elem(const neon_state *st, double *d, uint32_t r, uint32_t c) {
    return (double *)((char *)d + r * st->rowstride + (c >> 3) * 128) + (c & 7);
}

static int choose_shape(uint32_t H, uint32_t *R, uint32_t *C) {
    uint32_t best_r = 0, best_c = 0;
    double best = 1e30;
    for (uint32_t c = 16; c <= H / 4; c += 8) {
        if (H % c) continue;
        uint32_t r = H / c;
        uint8_t tmp[32];
        if (r % 4 || factorize(r, tmp) < 0 || factorize(c, tmp) < 0) continue;
        // pass-1 group: r * 128 bytes; pass-2 scratch: c * 32 bytes
        double cost = fabs(log2((double)r * 128) - log2((double)c * 32));
        if (r * 128 > 96 * 1024) cost += 20;
        if (c * 32 > 96 * 1024) cost += 20;
        if (cost < best) {
            best = cost;
            best_r = r;
            best_c = c;
        }
    }
    if (!best_r) return -1;
    *R = best_r;
    *C = best_c;
    return 0;
}

#define s_at ahn_s_at
#define weight_at ahn_weight

static int neon_init_impl(ahn_ctx *ctx, int sme);
static int neon_init(ahn_ctx *ctx) { return neon_init_impl(ctx, 0); }

static int sme_available(void) {
    int v = 0;
    size_t len = sizeof v;
    if (sysctlbyname("hw.optional.arm.FEAT_SME_F64F64", &v, &len, NULL, 0)) return 0;
    return v;
}

static int sme_init(ahn_ctx *ctx) {
    if (!sme_available()) return -1;
    return neon_init_impl(ctx, 1);
}

static int neon_init_impl(ahn_ctx *ctx, int sme) {
    if (ctx->L % 64) return -1;
    neon_state *st = calloc(1, sizeof *st);
    if (!st) return -1;
    ctx->bk = st;
    st->sme = sme;
    st->H = ctx->L / 2;
    if (choose_shape(st->H, &st->R, &st->C)) return -1;
    uint32_t R = st->R, C = st->C, H = st->H;
    st->ngroups = C / G;
    if (ctx->L - ctx->top_t > 2 * G) return -1;
    st->rowstride = (size_t)C * 16 + 128;
    if (nplan_init(&st->pr, R, 0) || nplan_init(&st->pc, C, 1)) return -1;
    for (uint32_t q = 0; q < C; q++) {
        if (st->pc.pos[C - 1 - st->pc.rev[q]] != C - 1 - q) return -1;
    }
    ctx->num_doubles = R * st->rowstride / 8;

    // four-step twiddles w_H^{c*k_r} at row position p
    st->t4 = ahn_alloc(ctx->num_doubles * sizeof(double));
    for (uint32_t p = 0; p < R; p++) {
        uint32_t kr = st->pr.rev[p];
        for (uint32_t c = 0; c < C; c++) {
            double a = 2.0 * (double)((uint64_t)c * kr % H) / H;
            double *e = elem(st, st->t4, p, c);
            e[0] = __cospi(a);
            e[IM] = -__sinpi(a);
        }
    }

    // compressed weights
    uint64_t b = UINT64_MAX;
    for (uint32_t j = 0; j < ctx->L; j++) {
        if (ctx->bits[j] < b) b = ctx->bits[j];
    }
    st->bsmall = (uint32_t)b;
    st->base_small = ldexp(1.0, (int)b);
    st->ibase_small = ldexp(1.0, -(int)b);
    st->scale = (double)ctx->k / (4.0 * H);
    for (uint32_t i = 0; i < 16; i++) {
        st->B[i] = weight_at(ctx, i, s_at(ctx, i));
        st->IB[i] = 1.0 / st->B[i];
        st->IBs[i] = st->scale / st->B[i];
    }
    st->nconsts = ahn_alloc(48 * sizeof(double));
    for (int i = 0; i < 16; i++) {
        st->nconsts[i] = st->B[i];
        st->nconsts[16 + i] = st->IBs[i];
    }
    st->nconsts[32] = st->nconsts[33] = st->base_small;
    st->nconsts[34] = st->nconsts[35] = st->ibase_small;
    {
        uint64_t k52 = 1ull << 52;
        memcpy(&st->nconsts[36], &k52, 8);
        memcpy(&st->nconsts[37], &k52, 8);
    }
    size_t nseg = (size_t)st->ngroups * R;
    st->A = ahn_alloc(nseg * sizeof(double));
    st->IA = ahn_alloc(nseg * sizeof(double));
    st->mask = ahn_alloc(nseg * sizeof(uint64_t));
    for (uint32_t g = 0; g < st->ngroups; g++) {
        for (uint32_t r = 0; r < R; r++) {
            uint64_t J = 2 * ((uint64_t)r * C + g * G);
            uint64_t sJ = s_at(ctx, J);
            size_t idx = (size_t)g * R + r;
            st->A[idx] = weight_at(ctx, J, sJ);
            st->IA[idx] = 1.0 / st->A[idx];
            uint64_t m = 0;
            for (uint32_t i = 0; i < 16; i++) {
                uint64_t sj = s_at(ctx, J + i);
                int64_t e = (int64_t)(sJ + s_at(ctx, i) - sj);
                if (e < 0 || e > 1) return -1;
                if (e) m |= 1ull << i;
                uint64_t bits = s_at(ctx, J + i + 1) - sj;
                if (bits != b && bits != b + 1) return -1;
                if (bits == b + 1) m |= 1ull << (16 + i);
            }
            st->mask[idx] = m;
        }
    }

    // table-driven normaliser for sizes where the extra traffic stays in cache
    if (H <= (1u << 17) && !getenv("AHN_NO_NTAB")) {
        st->ntab = ahn_alloc((size_t)st->ngroups * (R / 2) * 16 * 8 * sizeof(double));
        double *t = st->ntab;
        for (uint32_t g = 0; g < st->ngroups; g++) {
            for (uint32_t r = 0; r < R; r += 2) {
                for (uint32_t i = 0; i < 16; i++) {
                    for (uint32_t l = 0; l < 2; l++) {
                        size_t idx = (size_t)g * R + r + l;
                        uint64_t m = st->mask[idx];
                        double e2 = (m >> i) & 1 ? 0.5 : 1.0;
                        double base = (m >> (16 + i)) & 1 ? 2 * st->base_small : st->base_small;
                        t[0 + l] = st->IA[idx] * st->IBs[i] / e2;
                        t[2 + l] = st->A[idx] * st->B[i] * e2;
                        t[4 + l] = 1.0 / base;
                        t[6 + l] = base;
                    }
                    t += 8;
                }
            }
        }
    }

    // pass 2 tables
    st->partner = malloc(R * sizeof(uint32_t));
    st->rs = ahn_alloc(2 * R * sizeof(double));
    for (uint32_t p = 0; p < R; p++) {
        uint32_t kr = st->pr.rev[p];
        st->partner[p] = st->pr.pos[(R - kr) % R];
        st->rs[2 * p] = __cospi(2.0 * kr / H);
        st->rs[2 * p + 1] = -__sinpi(2.0 * kr / H);
    }
    st->cs = ahn_alloc(2 * C * sizeof(double));
    for (uint32_t q = 0; q < C; q++) {
        st->cs[2 * q] = __cospi(2.0 * st->pc.rev[q] / C);
        st->cs[2 * q + 1] = -__sinpi(2.0 * st->pc.rev[q] / C);
    }
    st->yscr = ahn_alloc((size_t)H * 16 + 64);
    st->pairs = malloc(R / 2 * sizeof *st->pairs);
    {
        uint32_t p0 = st->pr.pos[0], ph = st->pr.pos[R / 2], np = 0;
        st->pairs[np][0] = p0;
        st->pairs[np++][1] = ph;
        for (uint32_t p = 0; p < R; p++) {
            uint32_t pp = st->partner[p];
            if (p == p0 || p == ph || pp < p) continue;
            st->pairs[np][0] = p;
            st->pairs[np++][1] = pp;
        }
        st->npairs = np;
    }
    // threads: each needs at least two groups (the last range must hold the fold)
    int nth = ctx->threads > 64 ? 64 : (int)ctx->threads;
    if (nth > (int)(st->ngroups / 2)) nth = (int)(st->ngroups / 2);
    if (nth < 1) nth = 1;
    st->nth = nth;
    for (int t = 0; t <= nth; t++) st->gstart[t] = (uint32_t)((uint64_t)st->ngroups * t / nth);
    for (int t = 0; t < nth; t++) {
        st->scr[t] = ahn_alloc((size_t)C * 32);
        st->carry[t] = ahn_alloc(R * sizeof(double));
    }
    st->pool = nth > 1 ? ahn_pool_new(nth) : NULL;
    return 0;
}

static void neon_free(ahn_ctx *ctx) {
    neon_state *st = ctx->bk;
    nplan_free(&st->pr);
    nplan_free(&st->pc);
    ahn_free_aligned(st->t4);
    ahn_free_aligned(st->A);
    ahn_free_aligned(st->IA);
    ahn_free_aligned(st->mask);
    ahn_free_aligned(st->nconsts);
    ahn_free_aligned(st->ntab);
    free(st->partner);
    ahn_free_aligned(st->rs);
    ahn_free_aligned(st->cs);
    ahn_pool_free(st->pool);
    for (int t = 0; t < st->nth; t++) {
        ahn_free_aligned(st->scr[t]);
        ahn_free_aligned(st->carry[t]);
    }
    free(st->pairs);
    ahn_free_aligned(st->yscr);
    free(st);
    ctx->bk = NULL;
}

// ---- weights / digits helpers ---------------------------------------------

typedef struct {
    double w, iw, osc, base;
} dinfo;

static inline dinfo digit_info(const neon_state *st, uint32_t g, uint32_t r, uint32_t i) {
    size_t idx = (size_t)g * st->R + r;
    uint64_t m = st->mask[idx];
    dinfo d;
    double e2 = (m >> i) & 1 ? 0.5 : 1.0;
    d.w = st->A[idx] * st->B[i] * e2;
    d.iw = st->IA[idx] * st->IB[i] / e2;
    d.osc = st->IA[idx] * st->IBs[i] / e2;
    d.base = (m >> (16 + i)) & 1 ? 2 * st->base_small : st->base_small;
    return d;
}

// digit i of row r in group g lives at column g*8 + i/2, re (i even) / im
static inline double *dptr(const neon_state *st, double *d, uint32_t g, uint32_t r, uint32_t i) {
    return elem(st, d, r, g * G + i / 2) + ((i & 1) ? IM : 0);
}

static inline int sme_stage(const neon_state *st, const stage *s) {
    return st->sme && s->p == 8 && ((size_t)s->s * s->nblocks) % 2 == 0;
}

static void group_fwd(neon_state *st, double *d, uint32_t g) {
    double *base = (double *)((char *)d + g * 128);
    ptrdiff_t delta = (char *)st->t4 - (char *)d;
    if (!st->sme) {
        plan_fwd(&st->pr, 1, base, st->rowstride, 4, 1, delta);
        return;
    }
    size_t es = st->rowstride;
    for (uint32_t i = 0; i < st->pr.nst; i++) {
        const stage *s = &st->pr.st[i];
        size_t bs = (size_t)s->s * s->p * es;
        if (sme_stage(st, s)) {
            ahn_sme_r8(base, es, s->nblocks, bs, s->s, s->mdif);
            if (s->s == 1) ahn_cmul_b(base, es, st->R, delta);
        } else if (s->s == 1) {
            s->dif_t4[1](base, es, 4, s->nblocks, bs, delta);
        } else {
            s->tw_f[1](base, es, 4, s->nblocks, bs, s->s, s->tw);
        }
    }
}

static void group_inv(neon_state *st, double *d, uint32_t g) {
    double *base = (double *)((char *)d + g * 128);
    ptrdiff_t delta = (char *)st->t4 - (char *)d;
    if (!st->sme) {
        plan_inv(&st->pr, 1, base, st->rowstride, 4, 1, delta);
        return;
    }
    size_t es = st->rowstride;
    for (uint32_t i = st->pr.nst; i-- > 0;) {
        const stage *s = &st->pr.st[i];
        size_t bs = (size_t)s->s * s->p * es;
        if (sme_stage(st, s)) {
            if (s->s == 1) ahn_cmulc_b(base, es, st->R, delta);
            ahn_sme_r8(base, es, s->nblocks, bs, s->s, s->mdit);
        } else if (s->s == 1) {
            s->dit_t4[1](base, es, 4, s->nblocks, bs, delta);
        } else {
            s->dit_tw[1](base, es, 4, s->nblocks, bs, s->s, s->tw);
        }
    }
}

// ---- pass 1 -----------------------------------------------------------------

// mode 0: normal (carry in/out, reweight); 1: group 0 first visit (store
// digits unweighted, carry-in 0); 2: group 0 final (input digits, absorb carry,
// reweight).
static void norm_rows_c(neon_state *st, double *d, double *carr, uint32_t g, uint32_t r0,
                        uint32_t r1, int mode, double *maxerr) {
    for (uint32_t r = r0; r < r1; r++) {
        double carry = mode == 1 ? 0 : carr[r];
        for (uint32_t i = 0; i < 16; i++) {
            double *p = dptr(st, d, g, r, i);
            dinfo di = digit_info(st, g, r, i);
            double x;
            if (mode == 2) {
                x = *p;
            } else {
                double v = *p * di.osc;
                x = rint(v);
                double e = fabs(v - x);
                if (e > *maxerr) *maxerr = e;
            }
            x += carry;
            if (mode == 2 && i == 15) {
                carry = 0;
            } else {
                carry = rint(x / di.base);
                x -= carry * di.base;
            }
            *p = mode == 1 ? x : x * di.w;
        }
        carr[r] = carry;
    }
}

// Last two rows of the last group: normalise, fold the top, reweight.
static void norm_last_rows(ahn_ctx *ctx, neon_state *st, double *d, double *carr,
                           double *maxerr, double *wrapq) {
    uint32_t g = st->ngroups - 1, R = st->R;
    norm_rows_c(st, d, carr, g, R - 2, R - 1, 0, maxerr);
    // last row: round + carry, keep integer digits
    uint32_t r = R - 1;
    int64_t dig[16];
    double carry = carr[r];
    for (uint32_t i = 0; i < 16; i++) {
        dinfo di = digit_info(st, g, r, i);
        double v = *dptr(st, d, g, r, i) * di.osc;
        double x = rint(v);
        double e = fabs(v - x);
        if (e > *maxerr) *maxerr = e;
        x += carry;
        carry = rint(x / di.base);
        x -= carry * di.base;
        dig[i] = (int64_t)x;
    }
    // fold (digits L-16 .. L-1 are dig[0..15])
    uint32_t L = ctx->L, t = ctx->top_t, base = L - 16;
    __int128 Hh = (__int128)(int64_t)carry << (ctx->sL - ctx->top_st);
    uint64_t pos = 0;
    for (uint32_t j = t; j < L; j++) {
        Hh += (__int128)dig[j - base] << pos;
        pos += ctx->bits[j];
    }
    unsigned m = ctx->top_m;
    __int128 lo = Hh & (((__int128)1 << m) - 1);
    __int128 hi = Hh >> m;
    __int128 kk = (__int128)ctx->k;
    __int128 q = hi >= 0 ? hi / kk : -((-hi + kk - 1) / kk);
    __int128 top = lo + ((hi - q * kk) << m);
    __int128 span = kk << m;
    if (2 * top >= span) {
        top -= span;
        q += 1;
    }
    if (t < L) {
        for (uint32_t j = t; j < L - 1; j++) {
            unsigned b = ctx->bits[j];
            __int128 qq = (top + ((__int128)1 << (b - 1))) >> b;
            dig[j - base] = (int64_t)(top - (qq << b));
            top = qq;
        }
        dig[L - 1 - base] = (int64_t)top;
    }
    for (uint32_t i = 0; i < 16; i++) {
        dinfo di = digit_info(st, g, r, i);
        *dptr(st, d, g, r, i) = (double)dig[i] * di.w;
    }
    carr[r] = 0;
    *wrapq = (double)(int64_t)q;
}

typedef double (*norm_fn)(double *, size_t, size_t, double *, const double *, const double *,
                          const uint64_t *, const double *);

// mode 0/1/2 = a/b/c (see gen_norm)
static void norm_rows(ahn_ctx *ctx, neon_state *st, double *d, double *carr, uint32_t g,
                      uint32_t r0, uint32_t r1, int mode, double *maxerr) {
    static const norm_fn fns[3][2][2] = {
        {{ahn_norm_a, ahn_norm_ae}, {ahn_norm_at, ahn_norm_ate}},
        {{ahn_norm_b, ahn_norm_be}, {ahn_norm_bt, ahn_norm_bte}},
        {{ahn_norm_c, ahn_norm_c}, {ahn_norm_ct, ahn_norm_ct}},
    };
    size_t idx = (size_t)g * st->R + r0;
    double *gb = (double *)((char *)d + g * 128 + r0 * st->rowstride);
    int err = ctx->op_count % ctx->err_every == 0;
    double e;
    static const norm_fn qfns[3][2] = {
        {ahn_norm_aq, ahn_norm_aqe}, {ahn_norm_bq, ahn_norm_bqe}, {ahn_norm_cq, ahn_norm_cq}};
    if (st->ntab) {
        const double *t = st->ntab + ((size_t)g * (st->R / 2) + r0 / 2) * 128;
        size_t np = (r1 - r0) / 2, nq = np & ~(size_t)1;
        e = 0;
        if (nq) e = qfns[mode][err](gb, st->rowstride, nq, carr + r0, t, NULL, NULL, NULL);
        if (np & 1) {
            double e2 = fns[mode][1][err]((double *)((char *)gb + nq * 2 * st->rowstride),
                                          st->rowstride, 1, carr + r0 + 2 * nq,
                                          t + nq * 128, NULL, NULL, NULL);
            if (e2 > e) e = e2;
        }
    } else {
        e = fns[mode][0][err](gb, st->rowstride, (r1 - r0) / 2, carr + r0, st->A + idx,
                              st->IA + idx, st->mask + idx, st->nconsts);
    }
    if (e > *maxerr) *maxerr = e;
}

// Pass 1, phase A for thread t: the first group of the range is normalised
// with carry-in 0 and kept as unweighted digits; the others are completed.
static void pass1_a(void *arg, int t, int n) {
    neon_state *st = arg;
    ahn_ctx *ctx = st->ctx;
    double *d = st->d, *carr = st->carry[t], maxerr = 0;
    uint32_t R = st->R, NG = st->ngroups, g0 = st->gstart[t], g1 = st->gstart[t + 1];
    (void)n;
    group_inv(st, d, g0);
    memset(carr, 0, R * sizeof(double));  // mode b takes its carry-in from the array
    norm_rows(ctx, st, d, carr, g0, 0, R, 1, &maxerr);
    for (uint32_t g = g0 + 1; g < g1; g++) {
        group_inv(st, d, g);
        if (g == NG - 1) {
            norm_rows(ctx, st, d, carr, g, 0, R - 2, 0, &maxerr);
            norm_last_rows(ctx, st, d, carr, &maxerr, &st->wrapq);
        } else {
            norm_rows(ctx, st, d, carr, g, 0, R, 0, &maxerr);
        }
        group_fwd(st, d, g);
    }
    st->maxerr_t[t] = maxerr;
}

// Phase B: absorb the carries from the previous range into the first group.
// Thread 0 receives the last range's carries shifted down one row plus the
// wrapped top.
static void pass1_b(void *arg, int t, int n) {
    neon_state *st = arg;
    uint32_t R = st->R, g0 = st->gstart[t];
    double in[4096], *cin = R <= 4096 ? in : malloc(R * sizeof(double));
    if (t == 0) {
        const double *last = st->carry[n - 1];
        cin[0] = st->wrapq;
        memcpy(cin + 1, last, (R - 1) * sizeof(double));
    } else {
        memcpy(cin, st->carry[t - 1], R * sizeof(double));
    }
    double maxerr = 0;
    norm_rows(st->ctx, st, st->d, cin, g0, 0, R, 2, &maxerr);
    group_fwd(st, st->d, g0);
    if (cin != in) free(cin);
}

static void pass1(ahn_ctx *ctx, double *d) {
    neon_state *st = ctx->bk;
    st->ctx = ctx;
    st->d = d;
    ahn_pool_run(st->pool, pass1_a, st);
    ahn_pool_run(st->pool, pass1_b, st);
    for (int t = 0; t < st->nth; t++) {
        if (st->maxerr_t[t] > ctx->maxerr) ctx->maxerr = st->maxerr_t[t];
    }
}

// ---- pass 2 -----------------------------------------------------------------

// Real-split op on one pair: a = U_k, b = U_kb, w2 = w^{2k}; y = operand pair
// (or NULL to square).
static inline void split1(double *ar, double *ai, double *br, double *bi, double w2r, double w2i,
                          const double *y) {
    double er = *ar + *br, ei = *ai - *bi;
    double dr = *ar - *br, di = *ai + *bi;
    double or_ = di, oi = -dr;
    double Xr, Xi, Yr, Yi;
    if (!y) {
        double e2r = er * er - ei * ei, e2i = 2 * er * ei;
        double o2r = or_ * or_ - oi * oi, o2i = 2 * or_ * oi;
        Xr = e2r + (w2r * o2r - w2i * o2i);
        Xi = e2i + (w2r * o2i + w2i * o2r);
        Yr = 2 * (er * or_ - ei * oi);
        Yi = 2 * (er * oi + ei * or_);
    } else {
        double fr = y[0] + y[2], fi = y[1] - y[3];
        double gr = y[1] + y[3], gi = -(y[0] - y[2]);
        double oor = or_ * gr - oi * gi, ooi = or_ * gi + oi * gr;
        Xr = (er * fr - ei * fi) + (w2r * oor - w2i * ooi);
        Xi = (er * fi + ei * fr) + (w2r * ooi + w2i * oor);
        Yr = (er * gr - ei * gi) + (or_ * fr - oi * fi);
        Yi = (er * gi + ei * gr) + (or_ * fi + oi * fr);
    }
    *ar = Xr - Yi;
    *ai = Xi + Yr;
    *br = Xr + Yi;
    *bi = -Xi + Yr;
}

// scratch accessors: element q, lane l
#define SR(s, q, l) ((s)[4 * (q) + (l)])
#define SI(s, q, l) ((s)[4 * (q) + 2 + (l)])

// General row pair (p, pp != p, both k_r != 0, R/2): lane 0 at q pairs with
// lane 1 at C-1-q.
static void split_pair_c(neon_state *st, double *s, const double *y, uint32_t p, uint32_t pp) {
    uint32_t C = st->C;
    for (uint32_t q = 0; q < C; q++) {
        uint32_t qb = C - 1 - q;
        for (int l = 0; l < 2; l++) {
            uint32_t row = l ? pp : p;
            double rr = st->rs[2 * row], ri = st->rs[2 * row + 1];
            double cr = st->cs[2 * q], ci = st->cs[2 * q + 1];
            double w2r = rr * cr - ri * ci, w2i = rr * ci + ri * cr;
            if (l == 1 && 0) continue;
            // each unordered pair is visited twice; handle only q < qb per lane
            if (q > qb) continue;
            double y4[4];
            if (y) {
                y4[0] = SR(y, q, l);
                y4[1] = SI(y, q, l);
                y4[2] = SR(y, qb, 1 - l);
                y4[3] = SI(y, qb, 1 - l);
            }
            split1(&SR(s, q, l), &SI(s, q, l), &SR(s, qb, 1 - l), &SI(s, qb, 1 - l), w2r, w2i,
                   y ? y4 : NULL);
        }
    }
}

// Special pair: lane 0 = row with k_r = 0, lane 1 = row with k_r = R/2 (or the
// k_r = 0 row alone if R is odd -- not used since R % 4 == 0).
static void split_special_c(neon_state *st, double *s, const double *y, uint32_t p0,
                            uint32_t ph) {
    uint32_t C = st->C;
    for (uint32_t q = 0; q < C; q++) {
        // lane 0: k_r = 0, partner k_c -> -k_c
        uint32_t kc = st->pc.rev[q], qb = st->pc.pos[(C - kc) % C];
        if (qb >= q) {
            double w2r = st->cs[2 * q], w2i = st->cs[2 * q + 1];
            double y4[4];
            if (y) {
                y4[0] = SR(y, q, 0);
                y4[1] = SI(y, q, 0);
                y4[2] = SR(y, qb, 0);
                y4[3] = SI(y, qb, 0);
            }
            if (qb == q) {
                double br = SR(s, q, 0), bi = SI(s, q, 0);
                split1(&SR(s, q, 0), &SI(s, q, 0), &br, &bi, w2r, w2i, y ? y4 : NULL);
            } else {
                split1(&SR(s, q, 0), &SI(s, q, 0), &SR(s, qb, 0), &SI(s, qb, 0), w2r, w2i,
                       y ? y4 : NULL);
            }
        }
        // lane 1: k_r = R/2, partner q -> C-1-q in the same row
        uint32_t qc = C - 1 - q;
        if (qc > q) {
            double rr = st->rs[2 * ph], ri = st->rs[2 * ph + 1];
            double cr = st->cs[2 * q], ci = st->cs[2 * q + 1];
            double w2r = rr * cr - ri * ci, w2i = rr * ci + ri * cr;
            double y4[4];
            if (y) {
                y4[0] = SR(y, q, 1);
                y4[1] = SI(y, q, 1);
                y4[2] = SR(y, qc, 1);
                y4[3] = SI(y, qc, 1);
            }
            split1(&SR(s, q, 1), &SI(s, q, 1), &SR(s, qc, 1), &SI(s, qc, 1), w2r, w2i,
                   y ? y4 : NULL);
        }
    }
    (void)p0;
}

static void pass2_job(void *arg, int t, int n) {
    neon_state *st = arg;
    uint32_t C = st->C;
    double *d = st->d, *s = st->scr[t];
    uint32_t i0 = (uint32_t)((uint64_t)st->npairs * t / n);
    uint32_t i1 = (uint32_t)((uint64_t)st->npairs * (t + 1) / n);
    for (uint32_t i = i0; i < i1; i++) {
        uint32_t p = st->pairs[i][0], pp = st->pairs[i][1];
        double *rp = (double *)((char *)d + p * st->rowstride);
        double *rpp = (double *)((char *)d + pp * st->rowstride);
        // first stage fused with the transpose, then the rest of the row FFT
        const stage *s0 = &st->pc.st[0];
        ahn_dif_r4_rows(rp, rpp, s, C, s0->tw);
        for (uint32_t k = 1; k < st->pc.nst; k++) {
            const stage *sk = &st->pc.st[k];
            size_t bs = (size_t)sk->s * sk->p * 32;
            if (sk->s == 1) sk->notw_f[0](s, 32, 1, sk->nblocks, bs, 1, NULL);
            else sk->tw_f[0](s, 32, 1, sk->nblocks, bs, sk->s, sk->tw);
        }
        const double *y = st->mul ? st->yscr + (size_t)i * 4 * C : NULL;
        if (i == 0) {
            split_special_c(st, s, y, p, pp);
        } else if (!y) {
            double rs[4] = {st->rs[2 * p], st->rs[2 * pp], st->rs[2 * p + 1], st->rs[2 * pp + 1]};
            ahn_split_sq(s, C, rs, st->cs);
        } else {
            split_pair_c(st, s, y, p, pp);
        }
        for (uint32_t k = st->pc.nst; k-- > 1;) {
            const stage *sk = &st->pc.st[k];
            size_t bs = (size_t)sk->s * sk->p * 32;
            if (sk->s == 1) sk->dit_notw[0](s, 32, 1, sk->nblocks, bs, 1, NULL);
            else sk->dit_tw[0](s, 32, 1, sk->nblocks, bs, sk->s, sk->tw);
        }
        ahn_dit_r4_rows(rp, rpp, s, C, s0->tw);
    }
}

static void prep_job(void *arg, int t, int n) {
    neon_state *st = arg;
    uint32_t C = st->C;
    uint32_t i0 = (uint32_t)((uint64_t)st->npairs * t / n);
    uint32_t i1 = (uint32_t)((uint64_t)st->npairs * (t + 1) / n);
    for (uint32_t i = i0; i < i1; i++) {
        uint32_t p = st->pairs[i][0], pp = st->pairs[i][1];
        double *s = st->yscr + (size_t)i * 4 * C;
        ahn_rows_in((double *)((char *)st->d + p * st->rowstride),
                    (double *)((char *)st->d + pp * st->rowstride), s, C);
        plan_fwd(&st->pc, 0, s, 32, 1, 0, 0);
    }
}

static void pass2(ahn_ctx *ctx, double *d, int mul) {
    neon_state *st = ctx->bk;
    st->d = d;
    st->mul = mul;
    ahn_pool_run(st->pool, pass2_job, st);
}

static void prep_operand(ahn_ctx *ctx, const double *d) {
    neon_state *st = ctx->bk;
    st->d = (double *)d;
    ahn_pool_run(st->pool, prep_job, st);
}

// ---- ops ---------------------------------------------------------------------

static void neon_from_digits(ahn_ctx *ctx, ahn_num *x, const int64_t *dig) {
    neon_state *st = ctx->bk;
    memset(x->d, 0, ctx->num_doubles * sizeof(double));
    for (uint32_t g = 0; g < st->ngroups; g++) {
        for (uint32_t r = 0; r < st->R; r++) {
            for (uint32_t i = 0; i < 16; i++) {
                size_t j = 2 * ((size_t)r * st->C + g * G) + i;
                *dptr(st, x->d, g, r, i) = (double)dig[j] * digit_info(st, g, r, i).w;
            }
        }
        group_fwd(st, x->d, g);
    }
}

static void neon_to_digits(ahn_ctx *ctx, const ahn_num *x, int64_t *dig) {
    neon_state *st = ctx->bk;
    double *tmp = ahn_alloc(ctx->num_doubles * sizeof(double));
    memcpy(tmp, x->d, ctx->num_doubles * sizeof(double));
    for (uint32_t g = 0; g < st->ngroups; g++) {
        group_inv(st, tmp, g);
        for (uint32_t r = 0; r < st->R; r++) {
            for (uint32_t i = 0; i < 16; i++) {
                size_t j = 2 * ((size_t)r * st->C + g * G) + i;
                double v = *dptr(st, tmp, g, r, i) * digit_info(st, g, r, i).iw / st->R;
                dig[j] = (int64_t)rint(v);
            }
        }
    }
    ahn_free_aligned(tmp);
}

static void neon_square(ahn_ctx *ctx, ahn_num *x) {
    pass2(ctx, x->d, 0);
    pass1(ctx, x->d);
    ctx->op_count++;
}

static void neon_mul(ahn_ctx *ctx, ahn_num *x, const ahn_num *y) {
    prep_operand(ctx, y->d);
    pass2(ctx, x->d, 1);
    pass1(ctx, x->d);
    ctx->op_count++;
}

static void neon_mul_small(ahn_ctx *ctx, ahn_num *x, int64_t m) {
    int64_t *dig = ctx->tmpdig;
    neon_to_digits(ctx, x, dig);
    for (uint32_t j = 0; j < ctx->L; j++) dig[j] *= m;
    ahn_carry_digits(ctx, dig);
    neon_from_digits(ctx, x, dig);
}

const ahn_backend_ops ahn_neon_ops = {
    .name = "neon",
    .init = neon_init,
    .free = neon_free,
    .to_digits = neon_to_digits,
    .from_digits = neon_from_digits,
    .square = neon_square,
    .mul = neon_mul,
    .mul_small = neon_mul_small,
};

const ahn_backend_ops ahn_sme_ops = {
    .name = "sme",
    .init = sme_init,
    .free = neon_free,
    .to_digits = neon_to_digits,
    .from_digits = neon_from_digits,
    .square = neon_square,
    .mul = neon_mul,
    .mul_small = neon_mul_small,
};
