// Two-pass backend: the production data layout and pass structure.
//
// Real length-L weighted convolution via a complex FFT of length H = L/2 on
// u_j = a_{2j} + i*a_{2j+1}, decomposed four-step as H = R x C:
//   element j = c + C*r lives at row r, column c (row-major, rows contiguous)
//   pass 1: column FFTs of length R (8 columns at a time) + twiddle w_H^{c*k_r}
//   pass 2: row FFTs of length C, real-split pointwise square, inverse row FFT
//   pass 1 (inverse): twiddle, inverse column FFT, normalise/carry, forward.
// Numbers are stored between operations in the state after forward pass 1.
//
// Complex data is stored as SoA pairs: [re c, re c+1, im c, im c+1].
// FFTs are in-place DIF (forward, natural -> digit-reversed) and DIT (inverse,
// digit-reversed -> natural), unnormalised.
//
// Real split (for frequency index k with partner kb = H-k, w = e^{-2 pi i/L}):
//   e = U_k + conj(U_kb), o = -i (U_k - conj(U_kb))
//   4 V_k  = e^2 + w^{2k} o^2 + 2i e o
//   4 V_kb = conj(e^2 + w^{2k} o^2) + 2i conj(e o)
// where V is the length-H DFT of the packed cyclic convolution.
#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define G 8  // columns per pass-1 group

typedef struct {
    uint32_t n;
    uint32_t nstages;
    uint8_t radix[32];
    double *twr, *twi;  // per-stage twiddles, concatenated
    uint32_t *rev;      // rev[pos] = frequency index at position pos
    uint32_t *pos;      // pos[k] = position of frequency k
} plan;

typedef struct {
    uint32_t H, R, C, ngroups;
    plan pr, pc;
    // four-step twiddles w_H^{c*k_r}: indexed [p][c] as separate re/im
    double *t4r, *t4i;
    // real-split twiddles w_H^{k_r} per row position and w_C^{k_c} per column pos
    double *rsr, *rsi, *csr, *csi;
    uint32_t *partner;  // row position partner
    // scratch
    double *sre, *sim;  // R x G
    double *g0;         // group 0 digits, R x 2G
    int64_t *carry;     // R
    double *rre, *rim, *pre, *pim;  // row scratch (C each)
    double *yre, *yim;              // mul operand rows (H each)
    double scale_sq;                // 1/(4H)
} f2_state;

// ---- small DFTs ----------------------------------------------------------

static void dft_p(uint32_t p, double *xr, double *xi, int inv) {
    double yr[8], yi[8];
    for (uint32_t t = 0; t < p; t++) {
        double sr = 0, si = 0;
        for (uint32_t u = 0; u < p; u++) {
            double a = 2.0 * ((u * t) % p) / p;
            double cr = __cospi(a), ci = inv ? __sinpi(a) : -__sinpi(a);
            sr += xr[u] * cr - xi[u] * ci;
            si += xr[u] * ci + xi[u] * cr;
        }
        yr[t] = sr;
        yi[t] = si;
    }
    memcpy(xr, yr, p * sizeof(double));
    memcpy(xi, yi, p * sizeof(double));
}

// ---- plans -----------------------------------------------------------------

static int factorize(uint32_t n, uint8_t *radix) {
    int m = 0;
    while (n % 8 == 0 && n > 8) { radix[m++] = 8; n /= 8; }
    static const uint8_t pr[] = {8, 4, 2, 3, 5, 7};
    for (int i = 0; i < 6; i++) {
        while (n % pr[i] == 0) { radix[m++] = pr[i]; n /= pr[i]; }
    }
    return n == 1 ? m : -1;
}

static int plan_init(plan *pl, uint32_t n) {
    pl->n = n;
    int m = factorize(n, pl->radix);
    if (m < 0) return -1;
    pl->nstages = (uint32_t)m;
    // twiddles: stage with block size sz = p*s needs w_sz^{t*j}, t<p, j<s
    size_t tot = 0;
    uint32_t sz = n;
    for (int st = 0; st < m; st++) {
        tot += sz;
        sz /= pl->radix[st];
    }
    pl->twr = malloc(tot * sizeof(double));
    pl->twi = malloc(tot * sizeof(double));
    pl->rev = malloc(n * sizeof(uint32_t));
    pl->pos = malloc(n * sizeof(uint32_t));
    size_t off = 0;
    sz = n;
    for (int st = 0; st < m; st++) {
        uint32_t p = pl->radix[st], s = sz / p;
        for (uint32_t t = 0; t < p; t++) {
            for (uint32_t j = 0; j < s; j++) {
                double a = 2.0 * (double)((uint64_t)t * j % sz) / sz;
                pl->twr[off + t * s + j] = __cospi(a);
                pl->twi[off + t * s + j] = -__sinpi(a);
            }
        }
        off += sz;
        sz = s;
    }
    // pos(k) = (k mod p1) * (n/p1) + pos_sub(k / p1)
    for (uint32_t k = 0; k < n; k++) {
        uint32_t kk = k, base = 0, size = n;
        for (int st = 0; st < m; st++) {
            uint32_t p = pl->radix[st];
            size /= p;
            base += (kk % p) * size;
            kk /= p;
        }
        pl->pos[k] = base;
        pl->rev[base] = k;
    }
    return 0;
}

static void plan_free(plan *pl) {
    free(pl->twr);
    free(pl->twi);
    free(pl->rev);
    free(pl->pos);
}

// In-place DIF over `lanes` interleaved sequences: element e of lane l is at
// [e*lanes + l].
static void fft_dif(const plan *pl, double *re, double *im, uint32_t lanes) {
    uint32_t n = pl->n, sz = n;
    size_t off = 0;
    double xr[8], xi[8];
    for (uint32_t st = 0; st < pl->nstages; st++) {
        uint32_t p = pl->radix[st], s = sz / p;
        for (uint32_t b = 0; b < n; b += sz) {
            for (uint32_t j = 0; j < s; j++) {
                for (uint32_t l = 0; l < lanes; l++) {
                    for (uint32_t t = 0; t < p; t++) {
                        size_t i = (size_t)(b + j + t * s) * lanes + l;
                        xr[t] = re[i];
                        xi[t] = im[i];
                    }
                    dft_p(p, xr, xi, 0);
                    for (uint32_t t = 0; t < p; t++) {
                        double wr = pl->twr[off + t * s + j], wi = pl->twi[off + t * s + j];
                        size_t i = (size_t)(b + j + t * s) * lanes + l;
                        re[i] = xr[t] * wr - xi[t] * wi;
                        im[i] = xr[t] * wi + xi[t] * wr;
                    }
                }
            }
        }
        off += sz;
        sz = s;
    }
}

static void fft_dit_inv(const plan *pl, double *re, double *im, uint32_t lanes) {
    uint32_t n = pl->n;
    // stage offsets
    size_t offs[32];
    uint32_t sizes[32];
    size_t off = 0;
    uint32_t sz = n;
    for (uint32_t st = 0; st < pl->nstages; st++) {
        offs[st] = off;
        sizes[st] = sz;
        off += sz;
        sz /= pl->radix[st];
    }
    double xr[8], xi[8];
    for (uint32_t st = pl->nstages; st-- > 0;) {
        uint32_t p = pl->radix[st];
        sz = sizes[st];
        uint32_t s = sz / p;
        off = offs[st];
        for (uint32_t b = 0; b < n; b += sz) {
            for (uint32_t j = 0; j < s; j++) {
                for (uint32_t l = 0; l < lanes; l++) {
                    for (uint32_t t = 0; t < p; t++) {
                        double wr = pl->twr[off + t * s + j], wi = -pl->twi[off + t * s + j];
                        size_t i = (size_t)(b + j + t * s) * lanes + l;
                        xr[t] = re[i] * wr - im[i] * wi;
                        xi[t] = re[i] * wi + im[i] * wr;
                    }
                    dft_p(p, xr, xi, 1);
                    for (uint32_t t = 0; t < p; t++) {
                        size_t i = (size_t)(b + j + t * s) * lanes + l;
                        re[i] = xr[t];
                        im[i] = xi[t];
                    }
                }
            }
        }
    }
}

// ---- layout helpers --------------------------------------------------------

static inline size_t idx_re(const f2_state *st, uint32_t r, uint32_t c) {
    size_t e = (size_t)r * st->C + c;
    return (e >> 1) * 4 + (e & 1);
}

// Choose H = R*C with C % 8 == 0, C >= 16, R moderate.
static int choose_shape(uint32_t H, uint32_t *R, uint32_t *C) {
    uint32_t best_r = 0, best_c = 0;
    double best = 1e30;
    for (uint32_t c = 16; c <= H / 2; c += 8) {
        if (H % c) continue;
        uint32_t r = H / c;
        uint8_t tmp[32];
        if (factorize(r, tmp) < 0 || factorize(c, tmp) < 0) continue;
        // prefer r*G*16 bytes <= 64KB and rows <= 64KB, balanced
        double cost = fabs(log2((double)r) - log2((double)c) + 2);
        if (r > 512) cost += 10;
        if (c > 4096) cost += 10;
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

static int f2_init(ahn_ctx *ctx) {
    if (ctx->L % 32) return -1;
    f2_state *st = calloc(1, sizeof *st);
    if (!st) return -1;
    ctx->bk = st;
    st->H = ctx->L / 2;
    if (choose_shape(st->H, &st->R, &st->C)) return -1;
    if (2 * G < ctx->L - ctx->top_t) return -1;  // top region must fit the last group
    st->ngroups = st->C / G;
    if (plan_init(&st->pr, st->R) || plan_init(&st->pc, st->C)) return -1;
    uint32_t R = st->R, C = st->C, H = st->H;
    // pass 2 relies on pos(C-1-k) == C-1-pos(k)
    for (uint32_t q = 0; q < C; q++) {
        if (st->pc.pos[C - 1 - st->pc.rev[q]] != C - 1 - q) return -1;
    }
    st->t4r = malloc((size_t)H * sizeof(double));
    st->t4i = malloc((size_t)H * sizeof(double));
    for (uint32_t p = 0; p < R; p++) {
        uint32_t kr = st->pr.rev[p];
        for (uint32_t c = 0; c < C; c++) {
            double a = 2.0 * (double)((uint64_t)c * kr % H) / H;
            st->t4r[(size_t)p * C + c] = __cospi(a);
            st->t4i[(size_t)p * C + c] = -__sinpi(a);
        }
    }
    st->rsr = malloc(R * sizeof(double));
    st->rsi = malloc(R * sizeof(double));
    st->partner = malloc(R * sizeof(uint32_t));
    for (uint32_t p = 0; p < R; p++) {
        uint32_t kr = st->pr.rev[p];
        double a = 2.0 * kr / H;
        st->rsr[p] = __cospi(a);
        st->rsi[p] = -__sinpi(a);
        st->partner[p] = st->pr.pos[(R - kr) % R];
    }
    st->csr = malloc(C * sizeof(double));
    st->csi = malloc(C * sizeof(double));
    for (uint32_t q = 0; q < C; q++) {
        double a = 2.0 * st->pc.rev[q] / C;
        st->csr[q] = __cospi(a);
        st->csi[q] = -__sinpi(a);
    }
    st->sre = ahn_alloc((size_t)R * G * sizeof(double));
    st->sim = ahn_alloc((size_t)R * G * sizeof(double));
    st->g0 = ahn_alloc((size_t)R * 2 * G * sizeof(double));
    st->carry = ahn_alloc(R * sizeof(int64_t));
    st->rre = ahn_alloc(C * sizeof(double));
    st->rim = ahn_alloc(C * sizeof(double));
    st->pre = ahn_alloc(C * sizeof(double));
    st->pim = ahn_alloc(C * sizeof(double));
    st->yre = ahn_alloc((size_t)H * sizeof(double));
    st->yim = ahn_alloc((size_t)H * sizeof(double));
    st->scale_sq = 1.0 / (4.0 * H);
    ctx->num_doubles = ctx->L;
    return 0;
}

static void f2_free(ahn_ctx *ctx) {
    f2_state *st = ctx->bk;
    plan_free(&st->pr);
    plan_free(&st->pc);
    free(st->t4r);
    free(st->t4i);
    free(st->rsr);
    free(st->rsi);
    free(st->csr);
    free(st->csi);
    free(st->partner);
    ahn_free_aligned(st->sre);
    ahn_free_aligned(st->sim);
    ahn_free_aligned(st->g0);
    ahn_free_aligned(st->carry);
    ahn_free_aligned(st->rre);
    ahn_free_aligned(st->rim);
    ahn_free_aligned(st->pre);
    ahn_free_aligned(st->pim);
    ahn_free_aligned(st->yre);
    ahn_free_aligned(st->yim);
    free(st);
    ctx->bk = NULL;
}

// ---- pass 1 pieces -----------------------------------------------------------

static void group_load(const f2_state *st, const double *d, uint32_t g) {
    for (uint32_t r = 0; r < st->R; r++) {
        for (uint32_t l = 0; l < G; l++) {
            size_t i = idx_re(st, r, g * G + l);
            st->sre[r * G + l] = d[i];
            st->sim[r * G + l] = d[i + 2];
        }
    }
}

static void group_store(const f2_state *st, double *d, uint32_t g) {
    for (uint32_t r = 0; r < st->R; r++) {
        for (uint32_t l = 0; l < G; l++) {
            size_t i = idx_re(st, r, g * G + l);
            d[i] = st->sre[r * G + l];
            d[i + 2] = st->sim[r * G + l];
        }
    }
}

// inverse twiddle + inverse column FFT on the group in scratch
static void group_inverse(const f2_state *st, uint32_t g) {
    for (uint32_t p = 0; p < st->R; p++) {
        for (uint32_t l = 0; l < G; l++) {
            size_t t = (size_t)p * st->C + g * G + l;
            double wr = st->t4r[t], wi = -st->t4i[t];
            double xr = st->sre[p * G + l], xi = st->sim[p * G + l];
            st->sre[p * G + l] = xr * wr - xi * wi;
            st->sim[p * G + l] = xr * wi + xi * wr;
        }
    }
    fft_dit_inv(&st->pr, st->sre, st->sim, G);
}

static void group_forward(const f2_state *st, uint32_t g) {
    fft_dif(&st->pr, st->sre, st->sim, G);
    for (uint32_t p = 0; p < st->R; p++) {
        for (uint32_t l = 0; l < G; l++) {
            size_t t = (size_t)p * st->C + g * G + l;
            double wr = st->t4r[t], wi = st->t4i[t];
            double xr = st->sre[p * G + l], xi = st->sim[p * G + l];
            st->sre[p * G + l] = xr * wr - xi * wi;
            st->sim[p * G + l] = xr * wi + xi * wr;
        }
    }
}

// digit index of (row r, column c, re/im h)
static inline size_t digit_of(const f2_state *st, uint32_t r, uint32_t c, uint32_t h) {
    return 2 * ((size_t)r * st->C + c) + h;
}

// Round scratch (scaled) values of row r into 2G integer digits, carrying.
static int64_t row_round_carry(ahn_ctx *ctx, f2_state *st, uint32_t g, uint32_t r, double scale,
                               int64_t carry, int64_t *dig, double *maxerr) {
    for (uint32_t l = 0; l < G; l++) {
        for (uint32_t h = 0; h < 2; h++) {
            size_t di = digit_of(st, r, g * G + l, h);
            double v = (h ? st->sim : st->sre)[r * G + l] * scale * ctx->oscale[di];
            double rv = rint(v);
            double e = fabs(v - rv);
            if (e > *maxerr) *maxerr = e;
            int64_t x = (int64_t)rv + carry;
            unsigned b = ctx->bits[di];
            int64_t q = (x + (1ll << (b - 1))) >> b;
            dig[2 * l + h] = x - (q << b);
            carry = q;
        }
    }
    return carry;
}

static void row_weight_store(ahn_ctx *ctx, f2_state *st, uint32_t g, uint32_t r,
                             const int64_t *dig) {
    for (uint32_t l = 0; l < G; l++) {
        st->sre[r * G + l] = (double)dig[2 * l] * ctx->w[digit_of(st, r, g * G + l, 0)];
        st->sim[r * G + l] = (double)dig[2 * l + 1] * ctx->w[digit_of(st, r, g * G + l, 1)];
    }
}

// Add carry into the 2G digits starting at the group's first column; the last
// digit absorbs any remainder.
static void absorb_carry(const ahn_ctx *ctx, const f2_state *st, uint32_t r, uint32_t g,
                         int64_t *dig, int64_t carry) {
    for (uint32_t i = 0; i < 2 * G && carry; i++) {
        if (i == 2 * G - 1) {
            dig[i] += carry;
            return;
        }
        unsigned b = ctx->bits[digit_of(st, r, g * G + i / 2, i & 1)];
        int64_t x = dig[i] + carry;
        int64_t q = (x + (1ll << (b - 1))) >> b;
        dig[i] = x - (q << b);
        carry = q;
    }
}

// Fold for the top region, operating on the last group's last row digits.
static int64_t fold_last_row(const ahn_ctx *ctx, int64_t *dig, int64_t c) {
    // dig holds digits L-2G .. L-1
    uint32_t L = ctx->L, t = ctx->top_t, base = L - 2 * G;
    __int128 Hh = (__int128)c << (ctx->sL - ctx->top_st);
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
    if (t == L) return (int64_t)q;
    for (uint32_t j = t; j < L - 1; j++) {
        unsigned b = ctx->bits[j];
        __int128 qq = (top + ((__int128)1 << (b - 1))) >> b;
        dig[j - base] = (int64_t)(top - (qq << b));
        top = qq;
    }
    dig[L - 1 - base] = (int64_t)top;
    return (int64_t)q;
}

// Full pass 1 on data that has just finished pass 2.
static void pass1(ahn_ctx *ctx, double *d) {
    f2_state *st = ctx->bk;
    uint32_t R = st->R, NG = st->ngroups;
    double scale = st->scale_sq, maxerr = ctx->maxerr;
    int64_t dig[2 * G];
    // group 0: inverse, round, carry; keep digits for later
    group_load(st, d, 0);
    group_inverse(st, 0);
    for (uint32_t r = 0; r < R; r++) {
        st->carry[r] = row_round_carry(ctx, st, 0, r, scale, 0, dig, &maxerr);
        for (int i = 0; i < 2 * G; i++) st->g0[r * 2 * G + i] = (double)dig[i];
    }
    int64_t wrapq = 0;
    for (uint32_t g = 1; g < NG; g++) {
        group_load(st, d, g);
        group_inverse(st, g);
        for (uint32_t r = 0; r < R; r++) {
            int64_t c = row_round_carry(ctx, st, g, r, scale, st->carry[r], dig, &maxerr);
            if (g == NG - 1 && r == R - 1) {
                wrapq = fold_last_row(ctx, dig, c);
                c = 0;
            }
            st->carry[r] = c;
            row_weight_store(ctx, st, g, r, dig);
        }
        group_forward(st, g);
        group_store(st, d, g);
    }
    // group 0: absorb the carries from the last group (row r-1) and the wrap
    for (uint32_t r = 0; r < R; r++) {
        for (int i = 0; i < 2 * G; i++) dig[i] = (int64_t)st->g0[r * 2 * G + i];
        absorb_carry(ctx, st, r, 0, dig, r == 0 ? wrapq : st->carry[r - 1]);
        row_weight_store(ctx, st, 0, r, dig);
    }
    group_forward(st, 0);
    group_store(st, d, 0);
    ctx->maxerr = maxerr;
}

// ---- pass 2 ------------------------------------------------------------------

static void row_load(const f2_state *st, const double *d, uint32_t p, double *re, double *im) {
    for (uint32_t c = 0; c < st->C; c++) {
        size_t i = idx_re(st, p, c);
        re[c] = d[i];
        im[c] = d[i + 2];
    }
}

static void row_store(const f2_state *st, double *d, uint32_t p, const double *re,
                      const double *im) {
    for (uint32_t c = 0; c < st->C; c++) {
        size_t i = idx_re(st, p, c);
        d[i] = re[c];
        d[i + 2] = im[c];
    }
}

// One frequency pair (a = U_k, b = U_kb) with twiddle w2 = w^{2k}; squares
// (y == NULL) or multiplies by the transformed operand (ya, yb).
static inline void split_op(double *ar, double *ai, double *br, double *bi, double w2r,
                            double w2i, const double *y) {
    // e = a + conj(b), o = -i (a - conj(b))
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
        double eor = or_ * gr - oi * gi, eoi = or_ * gi + oi * gr;  // o_a * o_b
        Xr = (er * fr - ei * fi) + (w2r * eor - w2i * eoi);
        Xi = (er * fi + ei * fr) + (w2r * eoi + w2i * eor);
        // e_a o_b + o_a e_b
        Yr = (er * gr - ei * gi) + (or_ * fr - oi * fi);
        Yi = (er * gi + ei * gr) + (or_ * fi + oi * fr);
    }
    // V_k = X + iY ; V_kb = conj(X) + i conj(Y)
    *ar = Xr - Yi;
    *ai = Xi + Yr;
    *br = Xr + Yi;
    *bi = -Xi + Yr;
}

static void pass2(ahn_ctx *ctx, double *d, int mul) {
    f2_state *st = ctx->bk;
    uint32_t R = st->R, C = st->C;
    for (uint32_t p = 0; p < R; p++) {
        uint32_t pp = st->partner[p];
        if (pp < p) continue;
        double *are = st->rre, *aim = st->rim, *bre = st->pre, *bim = st->pim;
        row_load(st, d, p, are, aim);
        fft_dif(&st->pc, are, aim, 1);
        if (pp != p) {
            row_load(st, d, pp, bre, bim);
            fft_dif(&st->pc, bre, bim, 1);
        }
        uint32_t kr = st->pr.rev[p];
        double rr = st->rsr[p], ri = st->rsi[p];
        const double *yar = st->yre + (size_t)p * C, *yai = st->yim + (size_t)p * C;
        const double *ybr = st->yre + (size_t)pp * C, *ybi = st->yim + (size_t)pp * C;
        for (uint32_t q = 0; q < C; q++) {
            uint32_t qb;
            double *xr2, *xi2;
            if (kr == 0) {
                // partner of k_c is (C - k_c) mod C within the same row
                uint32_t kc = st->pc.rev[q];
                qb = st->pc.pos[(C - kc) % C];
                if (qb < q) continue;
                xr2 = &are[qb];
                xi2 = &aim[qb];
            } else if (pp == p) {
                qb = C - 1 - q;
                if (qb < q) continue;
                xr2 = &are[qb];
                xi2 = &aim[qb];
            } else {
                qb = C - 1 - q;
                xr2 = &bre[qb];
                xi2 = &bim[qb];
            }
            // w^{2k} = w_H^{k_r} * w_C^{k_c}
            double w2r = rr * st->csr[q] - ri * st->csi[q];
            double w2i = rr * st->csi[q] + ri * st->csr[q];
            double ar = are[q], ai = aim[q], br = *xr2, bi = *xi2;
            if (qb == q && pp == p) {
                // self-paired element
                double y4[4], *yp = NULL;
                if (mul) {
                    y4[0] = yar[q]; y4[1] = yai[q]; y4[2] = yar[q]; y4[3] = yai[q];
                    yp = y4;
                }
                split_op(&ar, &ai, &br, &bi, w2r, w2i, yp);
                are[q] = ar;
                aim[q] = ai;
                continue;
            }
            double y4[4], *yp = NULL;
            if (mul) {
                const double *pbr = (pp == p) ? yar : ybr, *pbi = (pp == p) ? yai : ybi;
                y4[0] = yar[q]; y4[1] = yai[q]; y4[2] = pbr[qb]; y4[3] = pbi[qb];
                yp = y4;
            }
            split_op(&ar, &ai, &br, &bi, w2r, w2i, yp);
            are[q] = ar;
            aim[q] = ai;
            *xr2 = br;
            *xi2 = bi;
        }
        fft_dit_inv(&st->pc, are, aim, 1);
        row_store(st, d, p, are, aim);
        if (pp != p) {
            fft_dit_inv(&st->pc, bre, bim, 1);
            row_store(st, d, pp, bre, bim);
        }
    }
}

// ---- backend ops -------------------------------------------------------------

static void f2_from_digits(ahn_ctx *ctx, ahn_num *x, const int64_t *dig) {
    f2_state *st = ctx->bk;
    for (uint32_t g = 0; g < st->ngroups; g++) {
        for (uint32_t r = 0; r < st->R; r++) {
            for (uint32_t l = 0; l < G; l++) {
                size_t d0 = digit_of(st, r, g * G + l, 0);
                st->sre[r * G + l] = (double)dig[d0] * ctx->w[d0];
                st->sim[r * G + l] = (double)dig[d0 + 1] * ctx->w[d0 + 1];
            }
        }
        group_forward(st, g);
        group_store(st, x->d, g);
    }
}

static void f2_to_digits(ahn_ctx *ctx, const ahn_num *x, int64_t *dig) {
    f2_state *st = ctx->bk;
    for (uint32_t g = 0; g < st->ngroups; g++) {
        group_load(st, x->d, g);
        group_inverse(st, g);
        for (uint32_t r = 0; r < st->R; r++) {
            for (uint32_t l = 0; l < G; l++) {
                size_t d0 = digit_of(st, r, g * G + l, 0);
                dig[d0] = (int64_t)rint(st->sre[r * G + l] / (st->R * ctx->w[d0]));
                dig[d0 + 1] = (int64_t)rint(st->sim[r * G + l] / (st->R * ctx->w[d0 + 1]));
            }
        }
    }
}

static void f2_square(ahn_ctx *ctx, ahn_num *x) {
    pass2(ctx, x->d, 0);
    pass1(ctx, x->d);
}

static void f2_mul(ahn_ctx *ctx, ahn_num *x, const ahn_num *y) {
    f2_state *st = ctx->bk;
    for (uint32_t p = 0; p < st->R; p++) {
        double *yr = st->yre + (size_t)p * st->C, *yi = st->yim + (size_t)p * st->C;
        row_load(st, y->d, p, yr, yi);
        fft_dif(&st->pc, yr, yi, 1);
    }
    pass2(ctx, x->d, 1);
    pass1(ctx, x->d);
}

static void f2_mul_small(ahn_ctx *ctx, ahn_num *x, int64_t m) {
    int64_t *dig = ctx->tmpdig;
    f2_to_digits(ctx, x, dig);
    for (uint32_t j = 0; j < ctx->L; j++) dig[j] *= m;
    ahn_carry_digits(ctx, dig);
    f2_from_digits(ctx, x, dig);
}

const ahn_backend_ops ahn_f2_ops = {
    .name = "c2pass",
    .init = f2_init,
    .free = f2_free,
    .to_digits = f2_to_digits,
    .from_digits = f2_from_digits,
    .square = f2_square,
    .mul = f2_mul,
    .mul_small = f2_mul_small,
};
