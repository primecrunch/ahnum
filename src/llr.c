// LLR (Morrison) test of N = k*2^n - 1, computed as PRST does (src/morrison.cpp, the
// "Morrison (LLR) test" with Q = -1), so primes, factors and RES64s agree with PRST.
//
// P is the smallest P >= 1 with Jacobi(P^2 + 4 | N) != 1; a common factor of 4P(P^2+4) and N
// is reported as a factor. Then V_k(P, -1) by a Lucas ladder, V_2k = V_k^2 + 2 (k odd), and
// n - 2 doublings V -> V^2 - 2 give V_{k*2^(n-1)}. N is prime iff that is 0; otherwise RES64
// is its low 64 bits.
//
// Error handling: no Gerbicz check applies to the V^2 - 2 chain. Round-off is checked on
// every step and a step over the limit is redone from the last verified state with a longer
// FFT. A non-zero result (composite) is confirmed by a second complete run at a different FFT
// length, and a third if they disagree, because an arithmetic error makes a prime look
// composite far more often than it lands exactly on 0.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "internal.h"
#include "kite.h"

static double now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((unsigned __int128)a * b % m);
}

// N mod m for N = k*2^n - 1
static uint64_t n_mod(uint64_t k, uint64_t n, uint64_t m) {
    uint64_t r = 1 % m, b = 2 % m;
    for (uint64_t e = n; e; e >>= 1, b = mulmod(b, b, m))
        if (e & 1) r = mulmod(r, b, m);
    return (mulmod(k % m, r, m) + m - 1 % m) % m;
}

static int jacobi_u64(uint64_t a, uint64_t m) {  // m odd
    int r = 1;
    a %= m;
    while (a) {
        while (!(a & 1)) {
            a >>= 1;
            if ((m & 7) == 3 || (m & 7) == 5) r = -r;
        }
        uint64_t t = a;
        a = m;
        m = t;
        if ((a & 3) == 3 && (m & 3) == 3) r = -r;
        a %= m;
    }
    return m == 1 ? r : 0;
}

// Jacobi(a | N) for small a, by reciprocity against N mod a.
static int jacobi_N(uint64_t a, uint64_t k, uint64_t n) {
    int r = 1;
    uint64_t nm8 = n_mod(k, n, 8);
    while (a && !(a & 1)) {
        a >>= 1;
        if (nm8 == 3 || nm8 == 5) r = -r;
    }
    if (a == 1) return r;
    if ((a & 3) == 3 && (nm8 & 3) == 3) r = -r;
    return r * jacobi_u64(n_mod(k, n, a), a);
}

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

int kite_llr_choose_P(uint64_t k, uint64_t n, uint32_t *P, uint64_t *factor) {
    uint32_t p = 1;
    while (jacobi_N((uint64_t)p * p + 4, k, n) == 1) {
        if (++p > 1000000) return -1;
    }
    *P = p;
    uint64_t d = 4ull * p * ((uint64_t)p * p + 4);
    uint64_t g = gcd_u64(d, n_mod(k, n, d));
    *factor = g > 1 ? g : 0;
    return 0;
}

// ---- checkpoint -------------------------------------------------------------------------

typedef struct {
    char magic[8];
    uint64_t k, n, P, iter, nlimbs, fft_len, run;
    uint64_t prev[2];  // RES64s of the earlier runs, for the composite confirmation
} llr_hdr;

static uint64_t fnv(const void *p, size_t len, uint64_t h) {
    const uint8_t *b = p;
    for (size_t i = 0; i < len; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static int ckpt_write(const char *path, const llr_hdr *h, const uint64_t *x) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    uint64_t s = fnv(h, sizeof *h, 0xcbf29ce484222325ull);
    s = fnv(x, h->nlimbs * 8, s);
    int ok = fwrite(h, sizeof *h, 1, f) == 1 && fwrite(x, 8, h->nlimbs, f) == h->nlimbs &&
             fwrite(&s, 8, 1, f) == 1;
    ok &= fclose(f) == 0;
    if (!ok || rename(tmp, path)) {
        remove(tmp);
        return -1;
    }
    return 0;
}

static int ckpt_read(const char *path, llr_hdr *h, uint64_t *x, size_t nl) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint64_t s = 0;
    int ok = fread(h, sizeof *h, 1, f) == 1 && !memcmp(h->magic, "KITELLR1", 8) &&
             h->nlimbs == nl && fread(x, 8, nl, f) == nl && fread(&s, 8, 1, f) == 1;
    fclose(f);
    if (!ok) return -1;
    uint64_t c = fnv(h, sizeof *h, 0xcbf29ce484222325ull);
    return fnv(x, nl * 8, c) == s ? 0 : -1;
}

int kite_llr_checkpoint_iter(const char *path, uint64_t k, uint64_t n, uint64_t *iter, int *run) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    llr_hdr h;
    int ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "KITELLR1", 8) && h.k == k &&
             h.n == n;
    fclose(f);
    if (!ok) return -1;
    *iter = h.iter;
    if (run) *run = (int)h.run;
    return 0;
}

// ---- one complete run ---------------------------------------------------------------------

// V_k(P, -1) into x (Lucas ladder on the bits of k).
static void lucas_vk(kite_ctx *ctx, uint64_t k, kite_num *x, kite_num *v1, kite_num *cP,
                     kite_num *c2) {
    // (v0, v1) = (V_m, V_m+1), from m = 0
    kite_set_u64(ctx, x, 2);
    kite_copy(ctx, v1, cP);
    uint64_t m = 0;
    int top = 63;
    while (top > 0 && !((k >> top) & 1)) top--;
    for (int b = top; b >= 0; b--) {
        int odd = (int)(m & 1);  // Q^m = -1 if m odd
        // V_m V_m+1 - P Q^m
        kite_num *t = kite_num_new(ctx);
        kite_copy(ctx, t, x);
        kite_mul(ctx, t, v1);
        if (odd) kite_add(ctx, t, cP);
        else kite_sub(ctx, t, cP);
        if ((k >> b) & 1) {
            // (V_2m+1, V_2m+2): V_2m+2 = V_m+1^2 - 2Q^(m+1)
            kite_square(ctx, v1);
            if (odd) kite_sub(ctx, v1, c2);  // Q^(m+1) = +1
            else kite_add(ctx, v1, c2);
            kite_copy(ctx, x, t);
            m = 2 * m + 1;
        } else {
            // (V_2m, V_2m+1): V_2m = V_m^2 - 2Q^m
            kite_square(ctx, x);
            if (odd) kite_add(ctx, x, c2);
            else kite_sub(ctx, x, c2);
            kite_copy(ctx, v1, t);
            m = 2 * m;
        }
        kite_num_free(ctx, t);
    }
}

static int is_zero(const uint64_t *l, size_t nl) {
    for (size_t i = 0; i < nl; i++)
        if (l[i]) return 0;
    return 1;
}

// Runs the doublings from the checkpoint (if any) to the end. Returns 0 and the final value in
// `out` (limbs), or a negative error.
static int llr_run(uint64_t k, uint64_t n, uint32_t P, const kite_llr_opts *o, uint32_t fft_len,
                   int run, const uint64_t *prev, uint64_t *out, kite_llr_result *res) {
    kite_opts fo = o->fft;
    if (fft_len) fo.fft_len = fft_len;
    double maxround = o->max_roundoff > 0 ? o->max_roundoff : 0.40;
    double ck_secs = o->checkpoint_secs > 0 ? o->checkpoint_secs : 600;
    uint64_t total = n - 1;  // doublings
    // testing hooks fire once per run, not again on a round-off redo
    int fault_due = o->inject_fault_at && ((o->inject_fault_runs ? o->inject_fault_runs : 1u) >> run & 1);
    int roundoff_due = run == 0 && (o->inject_roundoff_at || o->inject_roundoff_ladder);
    for (int attempt = 0;; attempt++) {
        kite_ctx *ctx = kite_ctx_new(k, n, -1, &fo);
        if (!ctx) return -1;
        kite_set_roundoff_check(ctx, 1);
        size_t nl = kite_limbs(ctx);
        uint64_t *vx = calloc(nl, 8);
        kite_num *x = kite_num_new(ctx), *v1 = kite_num_new(ctx), *cP = kite_num_new(ctx),
                 *c2 = kite_num_new(ctx);
        kite_set_u64(ctx, cP, P);
        kite_set_u64(ctx, c2, 2);
        uint64_t vi = 0;
        llr_hdr h;
        int rc = 0, round_fail = 0;
        int have_state = 0;  // vx holds a verified value at iteration vi
        if (o->checkpoint && ckpt_read(o->checkpoint, &h, vx, nl) == 0 && h.k == k && h.n == n &&
            h.P == P && h.run == (uint64_t)run && h.iter <= total) {
            vi = h.iter;
            if (!res->resumed) {
                res->resumed = 1;
                res->resumed_at = vi;
            }
            kite_set_limbs(ctx, x, vx);
            have_state = 1;
        } else {
            // The ladder's round-off is checked like a block's: a bad V_k would otherwise be
            // carried into the doublings, and with a checkpoint into every redo after them.
            kite_reset_roundoff(ctx);
            lucas_vk(ctx, k, x, v1, cP, c2);
            double e = kite_max_roundoff(ctx);
            if (e > res->maxerr) res->maxerr = e;
            if (e > maxround || (roundoff_due && o->inject_roundoff_ladder)) {
                roundoff_due = 0;
                round_fail = 1;
            } else {
                kite_get_limbs(ctx, x, vx);
                have_state = 1;
            }
        }
        double last_ck = now();
        uint64_t i = vi;
        const uint64_t B = 1000;  // iterations between in-memory verified states
        while (!round_fail && i < total) {
            kite_reset_roundoff(ctx);
            uint64_t stop = i + B < total ? i + B : total;
            for (; i < stop; i++) {
                kite_square(ctx, x);
                if (i == 0) kite_add(ctx, x, c2);  // V_2k = V_k^2 + 2 (k odd)
                else kite_sub(ctx, x, c2);
                if (fault_due && i + 1 == o->inject_fault_at) {
                    kite_debug_corrupt(ctx, x);
                    fault_due = 0;
                }
            }
            double e = kite_max_roundoff(ctx);
            if (e > res->maxerr) res->maxerr = e;
            if (e > maxround ||
                (roundoff_due && o->inject_roundoff_at && stop >= o->inject_roundoff_at)) {
                roundoff_due = 0;
                round_fail = 1;
                break;
            }
            kite_get_limbs(ctx, x, vx);
            vi = i;
            if (o->progress) o->progress(o->progress_arg, vi, total, e);
            int stop_now = o->abort_flag && *o->abort_flag;
            if (o->checkpoint && (stop_now || now() - last_ck > ck_secs)) {
                h = (llr_hdr){{'K', 'I', 'T', 'E', 'L', 'L', 'R', '1'}, k, n, P, vi, nl,
                              kite_fft_len(ctx), (uint64_t)run, {prev[0], prev[1]}};
                ckpt_write(o->checkpoint, &h, vx);
                last_ck = now();
            }
            if (stop_now) {
                rc = KITE_LLR_ABORTED;
                break;
            }
        }
        uint32_t L = kite_fft_len(ctx);
        if (!round_fail && rc == 0) {
            memcpy(out, vx, nl * 8);
            res->fft_len = L;
        }
        kite_num_free(ctx, x);
        kite_num_free(ctx, v1);
        kite_num_free(ctx, cP);
        kite_num_free(ctx, c2);
        kite_ctx_free(ctx);
        if (!round_fail) {
            free(vx);
            return rc;
        }
        // redo from the last verified state with a longer FFT (from the ladder if it failed)
        if (o->checkpoint && have_state) {
            h = (llr_hdr){{'K', 'I', 'T', 'E', 'L', 'L', 'R', '1'}, k, n, P, vi, nl, L, (uint64_t)run,
                          {prev[0], prev[1]}};
            ckpt_write(o->checkpoint, &h, vx);
        }
        free(vx);
        fo.fft_len = kite_next_fft_len(L + L / 25);
        res->roundoff_retries++;
        if (!fo.fft_len || attempt > 8) return -2;
    }
}

int kite_llr(uint64_t k, uint64_t n, const kite_llr_opts *opts, kite_llr_result *res) {
    kite_llr_opts o = opts ? *opts : (kite_llr_opts){0};
    memset(res, 0, sizeof *res);
    double t0 = now();
    if (n < 3 || !(k & 1)) return -1;
    uint32_t P;
    uint64_t factor;
    if (kite_llr_choose_P(k, n, &P, &factor)) return -1;
    res->P = P;
    if (factor) {
        res->factor = factor;
        res->seconds = now() - t0;
        return 0;
    }
    kite_ctx *probe = kite_ctx_new(k, n, -1, &o.fft);
    if (!probe) return -1;
    size_t nl = kite_limbs(probe);
    uint32_t L0 = kite_fft_len(probe);
    kite_ctx_free(probe);

    // Run 0 at the chosen length; a non-zero result is confirmed by run 1 at the next longer
    // length, and run 2 at the one after if they disagree. Results compare by RES64.
    // A run that ends in 0 can be outvoted: a prime whose run 0 went wrong is settled by runs 1
    // and 2 agreeing on 0, so the verdict comes from the agreeing run, not from run 0.
    uint64_t res64[3] = {0, 0, 0};
    int zero[3] = {0, 0, 0}, rc = 0, run = 0, final = 0;
    uint32_t L = 0;
    {  // resume the run sequence where a checkpoint left it
        llr_hdr h;
        uint64_t *tmp = calloc(nl, 8);
        if (o.checkpoint && ckpt_read(o.checkpoint, &h, tmp, nl) == 0 && h.k == k && h.n == n &&
            h.P == P && h.run < 3) {
            run = (int)h.run;
            res64[0] = h.prev[0];
            res64[1] = h.prev[1];
            // only RES64s are kept; a non-zero value with RES64 0 has probability 2^-64
            zero[0] = run > 0 && res64[0] == 0;
            zero[1] = run > 1 && res64[1] == 0;
            if (run > 0) L = (uint32_t)h.fft_len;
        }
        free(tmp);
    }
    uint64_t *out = calloc(nl, 8);
    for (; run < 3; run++) {
        if (run > 0) {
            res->confirm_runs = (uint32_t)run;
            if (o.on_confirm) o.on_confirm(o.progress_arg, run);
        }
        rc = llr_run(k, n, P, &o, L, run, res64, out, res);
        if (rc) break;
        res64[run] = out[0];
        zero[run] = is_zero(out, nl);
        if (run == 0 && zero[0]) break;  // prime
#define AGREE(a, b) (res64[a] == res64[b] && zero[a] == zero[b])
        if (run == 1 && AGREE(1, 0)) break;
        if (run == 2) {
            if (AGREE(2, 1)) final = 1;
            else if (AGREE(2, 0)) final = 0;
            else rc = -3;  // three different answers
            break;
        }
#undef AGREE
        uint32_t base = res->fft_len ? res->fft_len : L0;
        L = kite_next_fft_len(base + 1);
        if (!L) {
            rc = -2;
            break;
        }
    }
    free(out);
    if (rc == 0) {
        res->is_prime = zero[final];
        res->res64 = res64[final];
    }
    if (o.checkpoint && rc != KITE_LLR_ABORTED) remove(o.checkpoint);
    res->seconds = now() - t0;
    return rc;
}
