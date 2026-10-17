// Fermat PRP test of N = k*2^n - 1 with Gerbicz error checking, round-off
// fallback to a larger FFT, and checkpointing.
//
// Chain: x_0 = base^k, x_{i+1} = x_i^2, so x_n = base^(k*2^n) = base^(N+1).
// The residue is base^(N-1) = x_n / base^2 (mod N); N is a PRP iff
// x_n == base^2.
//
// Gerbicz check with block size B: d_t = prod_{j<=t} x_{jB} satisfies
// d_{t+1} = x_0 * d_t^(2^B). d is updated every B squarings; every M blocks
// the identity is verified with B extra squarings (overhead ~1/B + 1/M). A
// failed check rolls back to the last verified (i, x, d).
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Add 1 to one digit: a valid representation of a wrong value, which only
// the Gerbicz check can catch.
void kite_debug_corrupt(kite_ctx *ctx, kite_num *x) {
    int64_t *dig = malloc(ctx->L * sizeof(int64_t));
    ctx->ops->to_digits(ctx, x, dig);
    dig[ctx->L / 3] += 1;
    ctx->ops->from_digits(ctx, x, dig);
    free(dig);
}

// ---- small bigint helpers -------------------------------------------------

static uint64_t limbs_mod_small(const uint64_t *a, size_t n, uint64_t m) {
    unsigned __int128 r = 0;
    for (size_t i = n; i-- > 0;) r = ((r << 64) | a[i]) % m;
    return (uint64_t)r;
}

// a = a / d mod N (a < N, d small and coprime to N)
static void div_small_mod_N(uint64_t *a, const uint64_t *N, size_t n, uint64_t d) {
    uint64_t am = limbs_mod_small(a, n, d), nm = limbs_mod_small(N, n, d);
    uint64_t t = 0;
    while ((am + t * nm) % d) t++;
    unsigned __int128 c = 0;
    for (size_t i = 0; i < n; i++) {
        c += (unsigned __int128)a[i] + (unsigned __int128)N[i] * t;
        a[i] = (uint64_t)c;
        c >>= 64;
    }
    unsigned __int128 r = c;
    for (size_t i = n; i-- > 0;) {
        r = (r << 64) | a[i];
        a[i] = (uint64_t)(r / d);
        r %= d;
    }
}

// y = base^(N+1) mod N  ->  base^(N-1) mod N, also when gcd(base, N) > 1:
// write N = Q*M with Q built from the primes of base; the result is 0 mod Q
// and y / base^2 mod M, combined by CRT. Returns -1 if Q does not fit.
int kite_residue_nm1(uint64_t *y, const uint64_t *N, size_t n, uint64_t base) {
    uint64_t Q = 1, b = base;
    uint64_t *M = malloc((n + 1) * 8), *t = calloc(n + 1, 8);
    memcpy(M, N, n * 8);
    M[n] = 0;
    for (uint64_t p = 2; b > 1; p++) {
        if (b % p) continue;
        while (b % p == 0) b /= p;
        while (limbs_mod_small(M, n, p) == 0) {
            if (Q > (1ull << 40)) {
                free(M);
                free(t);
                return -1;
            }
            Q *= p;
            unsigned __int128 r = 0;
            for (size_t i = n; i-- > 0;) {
                r = (r << 64) | M[i];
                M[i] = (uint64_t)(r / p);
                r %= p;
            }
        }
    }
    if (Q == 1) {
        div_small_mod_N(y, N, n, base);
        div_small_mod_N(y, N, n, base);
        free(M);
        free(t);
        return 0;
    }
    // r = y mod M = (y*Q - q*N) / Q with q = floor(y*Q / N) < Q
    unsigned __int128 c = 0;
    for (size_t i = 0; i < n; i++) {
        c += (unsigned __int128)y[i] * Q;
        t[i] = (uint64_t)c;
        c >>= 64;
    }
    t[n] = (uint64_t)c;
    for (;;) {
        int ge = t[n] != 0;
        if (!ge) {
            ge = 1;
            for (size_t i = n; i-- > 0;) {
                if (t[i] != N[i]) {
                    ge = t[i] > N[i];
                    break;
                }
            }
        }
        if (!ge) break;
        uint64_t borrow = 0;
        for (size_t i = 0; i <= n; i++) {
            unsigned __int128 d = (unsigned __int128)t[i] - (i < n ? N[i] : 0) - borrow;
            t[i] = (uint64_t)d;
            borrow = (uint64_t)(d >> 64) & 1;
        }
    }
    unsigned __int128 r = 0;
    for (size_t i = n + 1; i-- > 0;) {
        r = (r << 64) | t[i];
        t[i] = (uint64_t)(r / Q);
        r %= Q;
    }
    // r / base^2 mod M (base is coprime to M)
    div_small_mod_N(t, M, n, base);
    div_small_mod_N(t, M, n, base);
    // x = t + M*u with x == 0 mod Q
    uint64_t rm = limbs_mod_small(t, n, Q), mm = limbs_mod_small(M, n, Q), u = 0;
    while ((rm + (unsigned __int128)mm * u) % Q) u++;
    c = 0;
    for (size_t i = 0; i < n; i++) {
        c += (unsigned __int128)t[i] + (unsigned __int128)M[i] * u;
        y[i] = (uint64_t)c;
        c >>= 64;
    }
    free(M);
    free(t);
    return 0;
}

// FNV-1a 64 over bytes
static uint64_t fnv(const void *p, size_t len, uint64_t h) {
    const uint8_t *b = p;
    for (size_t i = 0; i < len; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// ---- checkpoint -----------------------------------------------------------

typedef struct {
    char magic[8];
    uint64_t k, n, base, iter, nlimbs, fft_len;
} ckpt_hdr;

// Written as "KITECKP1". Releases up to 0.1.3 wrote "AHNCKPT1", the project's name before it was
// kite, while reading only "KITECKP1", so no Fermat test ever resumed. The layout is the same, so
// both are read and a test interrupted under 0.1.3 picks up where it was.
static const char CKPT_MAGIC[8] = {'K', 'I', 'T', 'E', 'C', 'K', 'P', '1'};
static const char CKPT_MAGIC_OLD[8] = {'A', 'H', 'N', 'C', 'K', 'P', 'T', '1'};

static int ckpt_magic_ok(const char magic[8]) {
    return !memcmp(magic, CKPT_MAGIC, 8) || !memcmp(magic, CKPT_MAGIC_OLD, 8);
}

static int ckpt_write(const char *path, uint64_t k, uint64_t n, uint32_t base, uint64_t iter,
                      uint32_t fft_len, const uint64_t *x, const uint64_t *d, size_t nl) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    ckpt_hdr h = {{0}, k, n, base, iter, nl, fft_len};
    memcpy(h.magic, CKPT_MAGIC, 8);
    uint64_t sum = fnv(&h, sizeof h, 0xcbf29ce484222325ull);
    sum = fnv(x, nl * 8, sum);
    sum = fnv(d, nl * 8, sum);
    int ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(x, 8, nl, f) == nl &&
             fwrite(d, 8, nl, f) == nl && fwrite(&sum, 8, 1, f) == 1;
    ok &= fclose(f) == 0;
    if (!ok || rename(tmp, path)) {
        remove(tmp);
        return -1;
    }
    return 0;
}

static int ckpt_read(const char *path, uint64_t k, uint64_t n, uint32_t base, uint64_t *iter,
                     uint32_t *fft_len, uint64_t *x, uint64_t *d, size_t nl) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    ckpt_hdr h;
    uint64_t sum = 0;
    int ok = fread(&h, sizeof h, 1, f) == 1 && ckpt_magic_ok(h.magic) && h.k == k &&
             h.n == n && h.base == base && h.nlimbs == nl && fread(x, 8, nl, f) == nl &&
             fread(d, 8, nl, f) == nl && fread(&sum, 8, 1, f) == 1;
    fclose(f);
    if (!ok) return -1;
    uint64_t s = fnv(&h, sizeof h, 0xcbf29ce484222325ull);
    s = fnv(x, nl * 8, s);
    s = fnv(d, nl * 8, s);
    if (s != sum) return -1;
    *iter = h.iter;
    *fft_len = (uint32_t)h.fft_len;
    return 0;
}

// ---- proof points -----------------------------------------------------------

// Points captured since the last verified state, reported once verified.
typedef struct {
    uint32_t *index;
    uint64_t **limbs;
    uint32_t count, cap;
} pending_points;

static void pending_clear(pending_points *p) {
    for (uint32_t i = 0; i < p->count; i++) free(p->limbs[i]);
    p->count = 0;
}

static void pending_add(pending_points *p, kite_ctx *ctx, uint32_t index, const kite_num *x) {
    if (p->count == p->cap) {
        p->cap = p->cap ? 2 * p->cap : 16;
        p->index = realloc(p->index, p->cap * sizeof *p->index);
        p->limbs = realloc(p->limbs, p->cap * sizeof *p->limbs);
    }
    p->index[p->count] = index;
    p->limbs[p->count] = malloc(kite_limbs(ctx) * 8);
    kite_get_limbs(ctx, x, p->limbs[p->count]);
    p->count++;
}

static int pending_flush(pending_points *p, const kite_prp_opts *o) {
    int rc = 0;
    for (uint32_t i = 0; i < p->count && !rc; i++)
        rc = o->on_point(o->point_arg, p->index[i], o->points[p->index[i]], p->limbs[i]);
    pending_clear(p);
    return rc;
}

int kite_prp_checkpoint_iter(const char *path, uint64_t k, uint64_t n, uint32_t base,
                            uint64_t *iter) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    ckpt_hdr h;
    int ok = fread(&h, sizeof h, 1, f) == 1 && ckpt_magic_ok(h.magic) && h.k == k &&
             h.n == n && h.base == base;
    fclose(f);
    if (!ok) return -1;
    *iter = h.iter;
    return 0;
}

// ---- driver ---------------------------------------------------------------

typedef struct {
    kite_ctx *ctx;
    kite_num *x, *d, *x0, *t;
} prp_state;

static void state_free(prp_state *s) {
    if (!s->ctx) return;
    kite_num_free(s->ctx, s->x);
    kite_num_free(s->ctx, s->d);
    kite_num_free(s->ctx, s->x0);
    kite_num_free(s->ctx, s->t);
    kite_ctx_free(s->ctx);
    memset(s, 0, sizeof *s);
}

static int state_new(prp_state *s, uint64_t k, uint64_t n, kite_opts o) {
    s->ctx = kite_ctx_new(k, n, -1, &o);
    if (!s->ctx) return -1;
    // sample round-off on every 4th operation: enough to spot an FFT length
    // that is too short; actual arithmetic errors are caught by Gerbicz
    kite_set_roundoff_check(s->ctx, 4);
    s->x = kite_num_new(s->ctx);
    s->d = kite_num_new(s->ctx);
    s->x0 = kite_num_new(s->ctx);
    s->t = kite_num_new(s->ctx);
    return s->x && s->d && s->x0 && s->t ? 0 : -1;
}

static int nums_equal(kite_ctx *ctx, const kite_num *a, const kite_num *b, uint64_t *la,
                      uint64_t *lb) {
    kite_get_limbs(ctx, a, la);
    kite_get_limbs(ctx, b, lb);
    return !memcmp(la, lb, kite_limbs(ctx) * 8);
}

// base^k mod N into limbs (exact power while it is below N, then reduce
// using the context arithmetic).
static void power_start(prp_state *s, uint64_t k, uint32_t base, uint64_t *l) {
    kite_ctx *ctx = s->ctx;
    size_t nl = kite_limbs(ctx);
    // left-to-right binary powering with small multiplications
    kite_set_u64(ctx, s->x0, 1);
    for (int b = 63; b >= 0; b--) {
        if (k >> b == 0) continue;
        kite_square(ctx, s->x0);
        if ((k >> b) & 1) kite_mul_small(ctx, s->x0, base);
    }
    // verify against an independent computation when base^k < N
    double bits = k * log2((double)base);
    if (bits + 2 < (double)ctx->n) {
        memset(l, 0, nl * 8);
        l[0] = 1;
        size_t used = 1;
        for (uint64_t i = 0; i < k; i++) {
            unsigned __int128 c = 0;
            for (size_t j = 0; j < used; j++) {
                c += (unsigned __int128)l[j] * base;
                l[j] = (uint64_t)c;
                c >>= 64;
            }
            if (c) l[used++] = (uint64_t)c;
        }
        kite_set_limbs(ctx, s->x0, l);
    }
}

int kite_prp(uint64_t k, uint64_t n, uint32_t base, const kite_prp_opts *opts,
            kite_prp_result *res) {
    kite_prp_opts o = opts ? *opts : (kite_prp_opts){0};
    memset(res, 0, sizeof *res);
    double t_start = now();
    uint32_t B = o.gerbicz_block;
    if (!B) {
        B = (uint32_t)sqrt((double)n);
        if (B < 32) B = 32;
        if (B > 2000) B = 2000;
    }
    uint32_t M = o.gerbicz_blocks ? o.gerbicz_blocks : (B / 4 > 8 ? B / 4 : 8);
    double maxround = o.max_roundoff > 0 ? o.max_roundoff : 0.40;
    double ck_secs = o.checkpoint_secs > 0 ? o.checkpoint_secs : 600;

    prp_state s = {0};
    if (state_new(&s, k, n, o.fft)) {
        state_free(&s);
        return -1;
    }
    size_t nl = kite_limbs(s.ctx);
    uint64_t *la = calloc(nl, 8), *lb = calloc(nl, 8);
    uint64_t *vx = calloc(nl, 8), *vd = calloc(nl, 8);  // last verified x, d
    uint64_t vi = 0;                                     // last verified iteration
    int rc = 0;
    pending_points pend = {0};

    power_start(&s, k, base, la);
    kite_get_limbs(s.ctx, s.x0, vx);
    memcpy(vd, vx, nl * 8);
    if (o.checkpoint) {
        uint32_t L = 0;
        if (!ckpt_read(o.checkpoint, k, n, base, &vi, &L, vx, vd, nl)) {
            res->resumed = 1;
            if (L > kite_fft_len(s.ctx)) {
                kite_opts fo = o.fft;
                fo.fft_len = L;
                state_free(&s);
                if (state_new(&s, k, n, fo)) {
                    rc = -1;
                    goto out;
                }
                power_start(&s, k, base, la);
            }
        }
    }
    uint64_t full = n / B * B;  // Gerbicz-protected part
    double last_ck = now();
    int consecutive_fail = 0;
    res->resumed_at = vi;
    uint32_t vp = 0;  // next point not yet reported
    if (o.on_point) {
        while (vp < o.npoints && o.points[vp] < vi + (vi > 0)) vp++;
        if (vp < o.npoints && o.points[vp] == 0) {  // x_0 = base^k, checked by power_start
            kite_set_limbs(s.ctx, s.x, vx);
            pending_add(&pend, s.ctx, vp++, s.x);
            if (pending_flush(&pend, &o)) {
                rc = KITE_PRP_POINT_FAILED;
                goto out;
            }
        }
    }
    for (;;) {
        // (re)start from the verified state
        kite_set_limbs(s.ctx, s.x, vx);
        kite_set_limbs(s.ctx, s.d, vd);
        kite_reset_roundoff(s.ctx);
        uint64_t i = vi;
        uint32_t pi = vp;
        int failed = 0, round_fail = 0;
        pending_clear(&pend);
        if (o.abort_flag && *o.abort_flag) {
            if (o.checkpoint) ckpt_write(o.checkpoint, k, n, base, vi, kite_fft_len(s.ctx), vx, vd, nl);
            rc = KITE_PRP_ABORTED;
            goto out;
        }
        // run up to B blocks, keep d before the last update in t
        uint64_t check_at = vi + (uint64_t)B * M;
        if (check_at > full) check_at = full;
        while (i < check_at) {
            // A round is B*M squarings, minutes at n in the millions, so a stop request is read
            // after every block rather than only between rounds. A caller that sends SIGTERM waits
            // a bounded time and then kills, losing everything since the last checkpoint.
            if (o.abort_flag && *o.abort_flag) break;
            for (uint32_t j = 0; j < B; j++) {
                kite_square(s.ctx, s.x);
                i++;
                if (o.inject_fault_at && i == o.inject_fault_at) {
                    kite_debug_corrupt(s.ctx, s.x);
                    o.inject_fault_at = 0;  // once
                }
                if (pi < o.npoints && o.points[pi] == i && o.on_point) pending_add(&pend, s.ctx, pi++, s.x);
            }
            if (i == check_at) kite_copy(s.ctx, s.t, s.d);
            kite_mul(s.ctx, s.d, s.x);
            if (kite_max_roundoff(s.ctx) > maxround) {
                round_fail = 1;
                break;
            }
            if (o.progress) o.progress(o.progress_arg, i, n, kite_max_roundoff(s.ctx));
        }
        // Stopped part way through the round: drop the unverified squarings. The check at the top
        // of the loop then saves the last verified state and returns.
        if (o.abort_flag && *o.abort_flag) continue;
        if (!round_fail && i > vi) {
            // verify d == x0 * t^(2^B)
            kite_square_n(s.ctx, s.t, B);
            kite_mul(s.ctx, s.t, s.x0);
            if (kite_max_roundoff(s.ctx) > maxround) round_fail = 1;
            else failed = !nums_equal(s.ctx, s.t, s.d, la, lb);
        }
        if (kite_max_roundoff(s.ctx) > res->maxerr) res->maxerr = kite_max_roundoff(s.ctx);
        if (round_fail) {
            // redo from the verified state with the next larger FFT length
            uint32_t L = kite_fft_len(s.ctx);
            kite_opts fo = o.fft;
            fo.fft_len = kite_next_fft_len(L + L / 25);
            state_free(&s);
            if (!fo.fft_len || state_new(&s, k, n, fo)) {
                rc = -2;
                goto out;
            }
            power_start(&s, k, base, la);
            res->roundoff_retries++;
            continue;
        }
        if (failed) {
            res->gerbicz_errors++;
            if (++consecutive_fail > 5) {
                rc = -3;
                goto out;
            }
            continue;
        }
        consecutive_fail = 0;
        kite_get_limbs(s.ctx, s.x, vx);
        kite_get_limbs(s.ctx, s.d, vd);
        vi = i;
        vp = pi;
        if (pend.count && pending_flush(&pend, &o)) {
            rc = KITE_PRP_POINT_FAILED;
            goto out;
        }
        if (o.checkpoint && (now() - last_ck > ck_secs || vi == full)) {
            ckpt_write(o.checkpoint, k, n, base, vi, kite_fft_len(s.ctx), vx, vd, nl);
            last_ck = now();
        }
        if (vi >= full) break;
    }
    // unprotected tail (< B squarings): compute twice and compare
    {
        uint64_t tail = n - full;
        for (int attempt = 0;; attempt++) {
            kite_set_limbs(s.ctx, s.x, vx);
            kite_set_limbs(s.ctx, s.t, vx);
            // stop at each point in the tail so it can be reported
            uint64_t i = full;
            uint32_t pi = vp;
            pending_clear(&pend);
            while (i < n) {
                uint64_t next = n;
                if (o.on_point && pi < o.npoints && o.points[pi] > i && o.points[pi] < next) next = o.points[pi];
                kite_square_n(s.ctx, s.x, next - i);
                kite_square_n(s.ctx, s.t, next - i);
                i = next;
                if (o.on_point && pi < o.npoints && o.points[pi] == i) pending_add(&pend, s.ctx, pi++, s.x);
            }
            if (nums_equal(s.ctx, s.x, s.t, la, lb)) {
                if (pend.count && pending_flush(&pend, &o)) {
                    rc = KITE_PRP_POINT_FAILED;
                    goto out;
                }
                break;
            }
            if (attempt > 5) {
                rc = -3;
                goto out;
            }
            res->gerbicz_errors++;
        }
    }
    // la = x_n; PRP iff x_n == base^2; residue = x_n / base^2
    {
        int is_prp = la[0] == (uint64_t)base * base;
        for (size_t j = 1; j < nl; j++) is_prp &= la[j] == 0;
        uint64_t *Nl = calloc(nl, 8);
        uint64_t sh = n % 64, lw = n / 64;
        Nl[lw] = k << sh;
        if (sh && lw + 1 < nl) Nl[lw + 1] = k >> (64 - sh);
        for (size_t j = 0; j < nl; j++) {
            if (Nl[j]--) break;
        }
        if (kite_residue_nm1(la, Nl, nl, base)) rc = -4;
        free(Nl);
        res->res64 = la[0];
        res->is_prp = is_prp;
    }
    res->fft_len = kite_fft_len(s.ctx);
    if (kite_max_roundoff(s.ctx) > res->maxerr) res->maxerr = kite_max_roundoff(s.ctx);
out:
    res->seconds = now() - t_start;
    pending_clear(&pend);
    free(pend.index);
    free(pend.limbs);
    state_free(&s);
    free(la);
    free(lb);
    free(vx);
    free(vd);
    return rc;
}
