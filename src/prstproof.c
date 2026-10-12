// PRST-compatible Fermat proofs. See prstproof.h and docs/prst-compat.md.
//
// Attribution: the proof construction, point layout, challenge hashing and file format follow
// PRST by Pavel Atnashev (https://github.com/patnashev/prst, src/proof.cpp, and
// https://github.com/patnashev/arithmetic, file.cpp), reimplemented in C so that certificates
// interoperate with PRST's. Line references are to PRST 14.1. The scheme is Pietrzak's
// verifiable delay function proof as used by PRST.
#include "prstproof.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "internal.h"
#include "md5.h"

#define PP_MAGIC 0x9f2b3cd4u
#define PP_APPID 4

void pp_giant_free(pp_giant *g) {
    free(g->w);
    g->w = NULL;
    g->len = 0;
}

static void giant_trim(pp_giant *g) {
    while (g->len > 0 && g->w[g->len - 1] == 0) g->len--;
}

// ---- fingerprints -----------------------------------------------------------

uint32_t pp_fingerprint(uint64_t k, uint64_t n) {
    const uint64_t p = 3417905339u;
    uint64_t r = 1, b = 2;
    for (uint64_t e = n; e; e >>= 1, b = b * b % p)
        if (e & 1) r = r * b % p;
    return (uint32_t)(((k % p) * r % p + p - 1) % p);
}

uint32_t pp_unique_fingerprint(uint32_t fp, const char *id) {  // file.cpp:182
    kite_md5 m;
    uint8_t d[16];
    kite_md5_init(&m);
    kite_md5_update(&m, &fp, 4);
    kite_md5_update(&m, id, strlen(id));
    kite_md5_final(&m, d);
    return d[0] | d[1] << 8 | d[2] << 16 | (uint32_t)d[3] << 24;
}

uint32_t pp_proof_fingerprint(uint64_t k, uint64_t n, uint32_t a) {  // fermat.cpp:269
    char id[48];
    snprintf(id, sizeof id, "%u.%llu", a, (unsigned long long)n);
    return pp_unique_fingerprint(pp_fingerprint(k, n), id);
}

// ---- files --------------------------------------------------------------------

typedef struct {
    uint8_t *p;
    size_t len, cap;
} buf;

static void put(buf *b, const void *p, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2;
        b->p = realloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, p, n);
    b->len += n;
}

static void put32(buf *b, uint32_t v) { put(b, &v, 4); }

static int write_all(const char *path, const void *p, size_t n) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    int ok = fwrite(p, 1, n, f) == n;
    ok &= fflush(f) == 0 && fsync(fileno(f)) == 0;
    ok &= fclose(f) == 0;
    if (!ok || rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

int pp_write(const char *path, int type, uint32_t fingerprint, int32_t iteration,
             const pp_giant *g, int ng) {  // file.cpp:198, TaskState::write
    buf b = {0};
    put32(&b, PP_MAGIC);
    put32(&b, PP_APPID | (uint32_t)(uint8_t)type << 16);
    if (fingerprint) put32(&b, fingerprint);
    put32(&b, (uint32_t)iteration);
    for (int i = 0; i < ng; i++) {
        put32(&b, (uint32_t)g[i].len);
        if (g[i].len) put(&b, g[i].w, 4 * (size_t)g[i].len);
    }
    char hex[33], mpath[4096];
    kite_md5_hex(b.p, b.len, hex);
    snprintf(mpath, sizeof mpath, "%s.md5", path);
    int rc = write_all(path, b.p, b.len) || write_all(mpath, hex, 32) ? -1 : 0;
    free(b.p);
    return rc;
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = n > 0 ? malloc((size_t)n) : NULL;
    if (p && fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p);
        p = NULL;
    }
    fclose(f);
    *len = p ? (size_t)n : 0;
    return p;
}

int pp_read(const char *path, int type, uint32_t fingerprint, int32_t *iteration, pp_giant *g,
            int maxg) {
    size_t len, mlen;
    uint8_t *p = read_file(path, &len);
    if (!p) return -1;
    char mpath[4096];
    snprintf(mpath, sizeof mpath, "%s.md5", path);
    uint8_t *m = read_file(mpath, &mlen);
    if (m) {  // file.cpp:236: a sidecar that is present must match
        char hex[33];
        kite_md5_hex(p, len, hex);
        int bad = mlen < 32 || memcmp(hex, m, 32) != 0;
        free(m);
        if (bad) {
            free(p);
            return -1;
        }
    }
    size_t off = 0;
    uint32_t v;
#define GET(x)                         \
    do {                               \
        if (off + 4 > len) goto fail;  \
        memcpy(&(x), p + off, 4);      \
        off += 4;                      \
    } while (0)
    GET(v);
    if (v != PP_MAGIC) goto fail;
    GET(v);
    if ((v & 0xff) != PP_APPID || ((v >> 16) & 0xff) != (uint32_t)type) goto fail;
    if (fingerprint) {
        GET(v);
        if (v != fingerprint) goto fail;
    }
    GET(*iteration);
    for (int i = 0; i < maxg; i++) {
        g[i].w = NULL;
        g[i].len = 0;
        if (off == len) continue;  // optional trailing giants (certificate a_power, a_base)
        int32_t gl;
        GET(gl);
        if (gl < 0) goto fail;  // only non-negative values occur here
        if (off + 4 * (size_t)gl > len) goto fail;
        g[i].len = gl;
        g[i].w = malloc(4 * (size_t)gl + 4);
        memcpy(g[i].w, p + off, 4 * (size_t)gl);
        off += 4 * (size_t)gl;
    }
#undef GET
    free(p);
    return 0;
fail:
    free(p);
    return -1;
}

static void remove_with_md5(const char *path) {
    char mpath[4096];
    snprintf(mpath, sizeof mpath, "%s.md5", path);
    remove(path);
    remove(mpath);
}

// ---- points -------------------------------------------------------------------

int pp_points(uint32_t iterations, int count, uint32_t *pos) {  // proof.cpp:160
    int M = 0;
    pos[0] = 0;
    for (int i = 1; i < count; i++) {
        M = (int)iterations;
        pos[i] = 0;
        for (int j = count / 2; j > 0 && (i & (j * 2 - 1)) != 0; j >>= 1) {
            M /= 2;
            if (i & j) pos[i] += (uint32_t)M;
            if (iterations & (uint32_t)(count / j / 2)) pos[i]++;
        }
    }
    pos[count] = iterations;
    return M;
}

// ---- numbers ------------------------------------------------------------------

void pp_from_num(kite_ctx *ctx, const kite_num *x, pp_giant *g) {
    size_t nl = kite_limbs(ctx);
    uint64_t *l = malloc(nl * 8);
    kite_get_limbs(ctx, x, l);
    g->w = malloc(nl * 8);
    for (size_t i = 0; i < nl; i++) {
        g->w[2 * i] = (uint32_t)l[i];
        g->w[2 * i + 1] = (uint32_t)(l[i] >> 32);
    }
    g->len = (int32_t)(2 * nl);
    giant_trim(g);
    free(l);
}

void pp_to_num(kite_ctx *ctx, const pp_giant *g, kite_num *x) {
    size_t nl = kite_limbs(ctx);
    uint64_t *l = calloc(nl, 8);
    for (int32_t i = 0; i < g->len && (size_t)i < 2 * nl; i++)
        l[i / 2] |= (uint64_t)g->w[i] << (32 * (i & 1));
    kite_set_limbs(ctx, x, l);
    free(l);
}

uint64_t pp_res64(const pp_giant *g) {
    uint64_t r = 0;
    if (g->len > 0) r = g->w[0];
    if (g->len > 1) r |= (uint64_t)g->w[1] << 32;
    return r;
}

static int giants_equal(const pp_giant *a, const pp_giant *b) {
    return a->len == b->len && (a->len == 0 || memcmp(a->w, b->w, 4 * (size_t)a->len) == 0);
}

// ---- challenges -----------------------------------------------------------------

// MD5(fp || Y || D), first 64 bits, made odd and free of prime factors below 1000
// (proof.cpp:511 hash_giants, :522 make_prime). The digest is read as a 2-word giant.
static unsigned __int128 challenge(uint32_t fp, const pp_giant *Y, const pp_giant *D) {
    kite_md5 m;
    uint8_t d[16];
    kite_md5_init(&m);
    kite_md5_update(&m, &fp, 4);
    if (Y->len) kite_md5_update(&m, Y->w, 4 * (size_t)Y->len);
    if (D->len) kite_md5_update(&m, D->w, 4 * (size_t)D->len);
    kite_md5_final(&m, d);
    uint32_t w0 = d[0] | d[1] << 8 | d[2] << 16 | (uint32_t)d[3] << 24;
    uint32_t w1 = d[4] | d[5] << 8 | d[6] << 16 | (uint32_t)d[7] << 24;
    w0 |= 1;
    if (w1 == 0) w1 = 1;
    unsigned __int128 g = (unsigned __int128)w1 << 32 | w0;
    for (;;) {
        int p, hit = 0;
        for (p = 2; p < 1000; p++) {
            int prime = 1;
            for (int q = 2; q * q <= p; q++)
                if (p % q == 0) {
                    prime = 0;
                    break;
                }
            if (prime && g % (unsigned)p == 0) {
                hit = 1;
                break;
            }
        }
        if (!hit) return g;
        g += 2;
    }
}

// X = X0^e (proof.cpp:538 exp_gw); X and X0 must differ.
static void power(kite_ctx *ctx, unsigned __int128 e, kite_num *X, const kite_num *X0) {
    int len = 0;
    while (len < 128 && (e >> len) > 1) len++;
    kite_copy(ctx, X, X0);
    for (int bit = len - 1; bit >= 0; bit--) {
        kite_square(ctx, X);
        if ((e >> bit) & 1) kite_mul(ctx, X, X0);
    }
}

static int roundoff_ok(kite_ctx *ctx) { return kite_max_roundoff(ctx) < 0.40; }

static int read_value(const char *prefix, const char *kind, int index, int type, uint32_t fp,
                      int32_t expect_iter, pp_giant *g) {
    char path[4096];
    int32_t it;
    snprintf(path, sizeof path, "%s.%s.%d", prefix, kind, index);
    if (pp_read(path, type, fp, &it, g, 1) != 0) {
        fprintf(stderr, "%s is missing or corrupt.\n", path);
        return -1;
    }
    if (it != expect_iter) {
        fprintf(stderr, "%s is missing or corrupt.\n", path);
        pp_giant_free(g);
        return -1;
    }
    return 0;
}

static int depth_of(int count) {
    int t = 0;
    while ((1 << t) < count) t++;
    return t;
}

// ---- save: compress the points into products (ProofSave::execute, proof.cpp:549) -----

int pp_save_products(kite_ctx *ctx, uint64_t k, uint64_t n, uint32_t a, int count,
                     const char *prefix, int keep, uint64_t *raw_res64) {
    uint32_t fpN = pp_fingerprint(k, n), fpP = pp_proof_fingerprint(k, n, a);
    int t = depth_of(count), rc = -1;
    uint32_t *pos = malloc((count + 1) * sizeof *pos);
    pp_points((uint32_t)n, count, pos);
    kite_num *Y = kite_num_new(ctx), *D = kite_num_new(ctx), *T = kite_num_new(ctx);
    kite_num **tree = calloc(t, sizeof *tree);
    unsigned __int128 *h = calloc(t, sizeof *h);
    pp_giant g = {0}, yg = {0}, dg = {0};
    for (int i = 0; i < t; i++) tree[i] = kite_num_new(ctx);
    kite_reset_roundoff(ctx);

    if (read_value(prefix, "proof", count, PP_TYPE_VALUE, fpP, (int32_t)pos[count], &g)) goto out;
    pp_to_num(ctx, &g, Y);
    pp_giant_free(&g);

    for (int i = 0; i < t; i++) {
        if (i == 0) {
            if (read_value(prefix, "proof", count / 2, PP_TYPE_VALUE, fpP, (int32_t)pos[count / 2], &g))
                goto out;
            pp_to_num(ctx, &g, D);
            pp_giant_free(&g);
        } else {
            for (int j = 0; j < (1 << i); j++) {
                int idx = (1 + j * 2) << (t - i - 1);
                if (read_value(prefix, "proof", idx, PP_TYPE_VALUE, fpP, (int32_t)pos[idx], &g)) goto out;
                pp_to_num(ctx, &g, D);
                pp_giant_free(&g);
                for (int kk = 1; kk <= i; kk++) {
                    if ((j & (1 << (kk - 1))) == 0) {
                        kite_copy(ctx, tree[i - kk], D);
                        break;
                    }
                    power(ctx, h[i - kk], T, tree[i - kk]);
                    kite_mul(ctx, D, T);
                }
            }
        }
        // the product, and the challenge from the committed Y and this product
        pp_from_num(ctx, D, &dg);
        char path[4096];
        snprintf(path, sizeof path, "%s.prod.%d", prefix, i);
        if (pp_write(path, PP_TYPE_PRODUCT, fpP, i, &dg, 1)) {
            fprintf(stderr, "Write to %s failed.\n", path);
            goto out;
        }
        pp_from_num(ctx, Y, &yg);
        h[i] = challenge(fpN, &yg, &dg);
        pp_giant_free(&yg);
        pp_giant_free(&dg);
        power(ctx, h[i], T, D);  // D^h
        kite_mul(ctx, Y, T);
        if (!roundoff_ok(ctx)) {
            fprintf(stderr, "round-off error while compressing proof points.\n");
            goto out;
        }
    }
    pp_from_num(ctx, Y, &yg);
    *raw_res64 = pp_res64(&yg);
    pp_giant_free(&yg);
    if (!keep) {
        for (int i = 1; i < count; i++) {
            char path[4096];
            snprintf(path, sizeof path, "%s.proof.%d", prefix, i);
            remove_with_md5(path);
        }
    }
    rc = 0;
out:
    for (int i = 0; i < t; i++) kite_num_free(ctx, tree[i]);
    free(tree);
    free(h);
    free(pos);
    kite_num_free(ctx, Y);
    kite_num_free(ctx, D);
    kite_num_free(ctx, T);
    return rc;
}

// ---- roots-of-unity check (proof.cpp:82, run in Proof::run before building) -----------

static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((unsigned __int128)a * b % m);
}

// (k*2^n - 2) mod m
static uint64_t nm1_mod(uint64_t k, uint64_t n, uint64_t m) {
    uint64_t r = 1 % m, b = 2 % m;
    for (uint64_t e = n; e; e >>= 1, b = mulmod(b, b, m))
        if (e & 1) r = mulmod(r, b, m);
    return (mulmod(k % m, r, m) + 2 * m - 2 % m) % m;
}

// For a composite result: R = a^(N-1) must not be a root of unity of the small order PRST
// tests. PRST (security 24) factors N-1 over 2 and the odd primes below 2^12, each as the
// largest power below 2^31, and fails if R^(product) == 1. Returns 0 if the check passes.
static int roots_of_unity_check(kite_ctx *ctx, uint64_t k, uint64_t n, uint32_t a,
                                const pp_giant *xn) {
    size_t nl = kite_limbs(ctx);
    uint64_t *l = calloc(nl, 8), *N = calloc(nl, 8);
    for (int32_t i = 0; i < xn->len && (size_t)i < 2 * nl; i++)
        l[i / 2] |= (uint64_t)xn->w[i] << (32 * (i & 1));
    N[n / 64] = k << (n % 64);
    if (n % 64 && n / 64 + 1 < nl) N[n / 64 + 1] = k >> (64 - n % 64);
    for (size_t j = 0; j < nl; j++)
        if (N[j]--) break;
    int rc = 0;
    if (kite_residue_nm1(l, N, nl, a) == 0) {
        int one = l[0] == 1;
        for (size_t j = 1; j < nl; j++) one &= l[j] == 0;
        if (!one) {  // a PRP result is not checked
            kite_num *R = kite_num_new(ctx), *X = kite_num_new(ctx), *T = kite_num_new(ctx);
            kite_set_limbs(ctx, R, l);
            kite_copy(ctx, X, R);
            // the power of 2 in N-1 = 2(k*2^(n-1) - 1) is 2^1
            kite_square(ctx, X);
            for (uint32_t p = 3; p < 4096; p += 2) {
                int prime = 1;
                for (uint32_t q = 3; q * q <= p; q += 2)
                    if (p % q == 0) {
                        prime = 0;
                        break;
                    }
                if (!prime || nm1_mod(k, n, p) != 0) continue;
                uint64_t d = p;  // largest p^e dividing N-1 with p^e < 2^31
                while (d * p < 2147483647u && nm1_mod(k, n, d * p) == 0) d *= p;
                kite_copy(ctx, T, X);  // X = X^d
                int len = 0;
                while ((d >> len) > 1) len++;
                for (int b = len - 1; b >= 0; b--) {
                    kite_square(ctx, X);
                    if ((d >> b) & 1) kite_mul(ctx, X, T);
                }
            }
            kite_get_limbs(ctx, X, l);
            one = l[0] == 1;
            for (size_t j = 1; j < nl; j++) one &= l[j] == 0;
            if (one) rc = -1;
            kite_num_free(ctx, R);
            kite_num_free(ctx, X);
            kite_num_free(ctx, T);
        }
    }
    free(l);
    free(N);
    return rc;
}

// ---- build: the certificate (ProofBuild::execute, proof.cpp:660, not Li, no security) ---

int pp_build_cert(kite_ctx *ctx, uint64_t k, uint64_t n, uint32_t a, int count,
                  const char *prefix, int keep, uint64_t *cert_res64) {
    uint32_t fpN = pp_fingerprint(k, n), fpP = pp_proof_fingerprint(k, n, a);
    int t = depth_of(count), rc = -1;
    uint32_t *pos = malloc((count + 1) * sizeof *pos);
    pp_points((uint32_t)n, count, pos);
    kite_num *X = kite_num_new(ctx), *Y = kite_num_new(ctx), *D = kite_num_new(ctx),
            *T = kite_num_new(ctx);
    pp_giant g = {0}, yg = {0}, dg = {0};
    kite_reset_roundoff(ctx);

    // point 0 must be a^k (proof.cpp:219)
    if (read_value(prefix, "proof", 0, PP_TYPE_VALUE, fpP, 0, &g)) goto out;
    pp_to_num(ctx, &g, X);
    {
        kite_set_u64(ctx, T, a);
        kite_set_u64(ctx, D, 1);
        for (int b = 63; b >= 0; b--) {
            kite_square(ctx, D);
            if ((k >> b) & 1) kite_mul(ctx, D, T);
        }
        pp_from_num(ctx, D, &dg);
        int ok = giants_equal(&g, &dg);
        pp_giant_free(&dg);
        pp_giant_free(&g);
        if (!ok) {
            fprintf(stderr, "invalid a^k.\n");
            goto out;
        }
    }
    if (read_value(prefix, "proof", count, PP_TYPE_VALUE, fpP, (int32_t)pos[count], &g)) goto out;
    pp_to_num(ctx, &g, Y);
    if (roots_of_unity_check(ctx, k, n, a, &g)) {
        fprintf(stderr, "roots of unity check failed.\n");
        pp_giant_free(&g);
        goto out;
    }
    pp_giant_free(&g);

    uint32_t M = pos[count];
    for (int i = 0; i < t; i++, M >>= 1) {
        if (read_value(prefix, "prod", i, PP_TYPE_PRODUCT, fpP, i, &dg)) goto out;
        pp_to_num(ctx, &dg, D);
        pp_from_num(ctx, Y, &yg);
        unsigned __int128 h = challenge(fpN, &yg, &dg);
        pp_giant_free(&yg);
        pp_giant_free(&dg);
        unsigned __int128 e = h * (M % 2 ? 2 : 1);  // exp *= b for odd M (b = 2)
        power(ctx, e, T, X);
        kite_copy(ctx, X, T);
        kite_mul(ctx, X, D);
        power(ctx, h, T, D);
        kite_mul(ctx, Y, T);
        if (!roundoff_ok(ctx)) {
            fprintf(stderr, "round-off error while building the certificate.\n");
            goto out;
        }
    }
    pp_from_num(ctx, Y, &yg);
    if (yg.len == 0) {
        fprintf(stderr, "invalid proof, the certificate is zero.\n");
        goto out;
    }
    *cert_res64 = pp_res64(&yg);
    pp_giant_free(&yg);
    {
        pp_giant c[3] = {{0}, {0}, {0}};  // X, a_power = 0, a_base = 0
        pp_from_num(ctx, X, &c[0]);
        char path[4096];
        snprintf(path, sizeof path, "%s.cert", prefix);
        int w = pp_write(path, PP_TYPE_CERT, fpN, (int32_t)M, c, 3);
        pp_giant_free(&c[0]);
        if (w) {
            fprintf(stderr, "Write to %s failed.\n", path);
            goto out;
        }
    }
    if (!keep) {
        char path[4096];
        snprintf(path, sizeof path, "%s.proof.0", prefix);
        remove_with_md5(path);
        snprintf(path, sizeof path, "%s.proof.%d", prefix, count);
        remove_with_md5(path);
        for (int i = 0; i < t; i++) {
            snprintf(path, sizeof path, "%s.prod.%d", prefix, i);
            remove_with_md5(path);
        }
    }
    rc = 0;
out:
    free(pos);
    kite_num_free(ctx, X);
    kite_num_free(ctx, Y);
    kite_num_free(ctx, D);
    kite_num_free(ctx, T);
    return rc;
}

// ---- verify: X^(2^M) with a Gerbicz check -----------------------------------------------

int pp_verify_cert(kite_ctx *ctx, uint64_t k, uint64_t n, const char *path, uint64_t *cert_res64,
                   uint32_t *Mout) {
    pp_giant c[3];
    int32_t M;
    if (pp_read(path, PP_TYPE_CERT, pp_fingerprint(k, n), &M, c, 3) != 0 || M < 0) {
        fprintf(stderr, "Invalid certificate file.\n");
        return -1;
    }
    int li = c[1].len != 0;
    pp_giant_free(&c[1]);
    pp_giant_free(&c[2]);
    if (li) {  // Gerbicz-Li certificates come from non-smooth tests, out of scope here
        pp_giant_free(&c[0]);
        fprintf(stderr, "Gerbicz-Li certificates are not supported.\n");
        return -1;
    }
    kite_num *x0 = kite_num_new(ctx), *x = kite_num_new(ctx), *d = kite_num_new(ctx),
            *e = kite_num_new(ctx);
    pp_to_num(ctx, &c[0], x0);
    pp_giant_free(&c[0]);
    uint32_t B = (uint32_t)sqrt((double)M);
    if (B < 1) B = 1;
    int rc = -1;
    for (int attempt = 0; attempt < 3 && rc; attempt++) {
        kite_reset_roundoff(ctx);
        // d_t = prod_{j<=t} x_{jB}; check d_T == d_{T-1}^(2^B) * x_0 once at the end
        kite_copy(ctx, x, x0);
        kite_copy(ctx, d, x0);
        uint32_t blocks = (uint32_t)M / B;
        for (uint32_t b = 0; b < blocks; b++) {
            if (b + 1 == blocks) kite_copy(ctx, e, d);  // d_{T-1}
            kite_square_n(ctx, x, B);
            kite_mul(ctx, d, x);
        }
        if (blocks) {
            kite_square_n(ctx, e, B);
            kite_mul(ctx, e, x0);
            pp_giant g1, g2;
            pp_from_num(ctx, d, &g1);
            pp_from_num(ctx, e, &g2);
            int ok = giants_equal(&g1, &g2);
            pp_giant_free(&g1);
            pp_giant_free(&g2);
            if (!ok) continue;
        }
        uint32_t tail = (uint32_t)M - blocks * B;
        kite_copy(ctx, e, x);
        kite_square_n(ctx, x, tail);
        kite_square_n(ctx, e, tail);
        pp_giant g1, g2;
        pp_from_num(ctx, x, &g1);
        pp_from_num(ctx, e, &g2);
        int ok = giants_equal(&g1, &g2) && roundoff_ok(ctx);
        if (ok) {
            *cert_res64 = pp_res64(&g1);
            rc = 0;
        }
        pp_giant_free(&g1);
        pp_giant_free(&g2);
    }
    if (Mout) *Mout = (uint32_t)M;
    kite_num_free(ctx, x0);
    kite_num_free(ctx, x);
    kite_num_free(ctx, d);
    kite_num_free(ctx, e);
    return rc;
}
