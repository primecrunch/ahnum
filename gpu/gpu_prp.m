// Prototype: Gerbicz-checked Fermat PRP test of N = k * 2^n - 1 on the GPU,
// B copies at once (every slot runs the same number; the point is throughput
// and an end-to-end check against known residues). Mirrors ahn_prp (src/prp.c):
//   x_0 = 3^k, x_{i+1} = x_i^2, x_n = 3^(N+1); residue = x_n / 9 = 3^(N-1).
//   Gerbicz, block size L: d_t = prod x_(jL), d_(t+1) = x_0 * d_t^(2^L); d is
//   updated every L squarings and the identity is verified every M blocks; a
//   failed check rolls back to the last verified (i, x, d). The final < L
//   squarings run twice and are compared.
// Values on the GPU are in Montgomery form X = x 2^n; squarings and products
// map Montgomery forms to Montgomery forms, so the Gerbicz identity holds as
// is. Comparisons reduce mod N on the CPU. The arithmetic is exact (integer
// NTT), so there is no round-off check; Gerbicz covers hardware faults and
// bugs.
//
// build: clang -O2 -fobjc-arc -Iinclude -Isrc gpu/gpu_prp.m build/libahnum.a \
//            -framework Metal -framework Foundation -o build/gpu_prp
// usage: build/gpu_prp K N [B] [--expect HEX] [--ckpt FILE] [--ckpt-secs S]
//                      [--fault ITER] [--block L]   (run from the repo root)
#include "gpu_ctx.h"

#include <math.h>

#include "ahnum.h"
#include "internal.h"

static double now_s(void) { return [[NSDate date] timeIntervalSince1970]; }

typedef struct {
    char magic[8];
    uint64_t k, n, B, W, vi;
} ckhdr;

static int ck_write(const char *path, ckhdr h, const uint64_t *vx, const uint64_t *vd, size_t words) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    int ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(vx, 8, words, f) == words &&
             fwrite(vd, 8, words, f) == words;
    ok &= fclose(f) == 0;
    return ok && rename(tmp, path) == 0 ? 0 : -1;
}

static int ck_read(const char *path, ckhdr *want, uint64_t *vx, uint64_t *vd, size_t words) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    ckhdr h;
    int ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, want->magic, 8) && h.k == want->k &&
             h.n == want->n && h.B == want->B && h.W == want->W && fread(vx, 8, words, f) == words &&
             fread(vd, 8, words, f) == words;
    fclose(f);
    if (ok) want->vi = h.vi;
    return ok ? 0 : -1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s K N [B] [--expect HEX] [--ckpt FILE] [--ckpt-secs S] "
                            "[--fault ITER] [--block L]\n",
                    argv[0]);
            return 2;
        }
        uint64_t k = strtoull(argv[1], 0, 10), n = strtoull(argv[2], 0, 10);
        uint32_t B = 64, L = 0;
        const char *expect = NULL, *ckpath = NULL;
        double ck_secs = 300;
        uint64_t fault_at = 0;
        int ai = 3;
        if (ai < argc && argv[ai][0] != '-') B = (uint32_t)atoi(argv[ai++]);
        for (; ai + 1 < argc; ai += 2) {
            if (!strcmp(argv[ai], "--expect")) expect = argv[ai + 1];
            else if (!strcmp(argv[ai], "--ckpt")) ckpath = argv[ai + 1];
            else if (!strcmp(argv[ai], "--ckpt-secs")) ck_secs = atof(argv[ai + 1]);
            else if (!strcmp(argv[ai], "--fault")) fault_at = strtoull(argv[ai + 1], 0, 10);
            else if (!strcmp(argv[ai], "--block")) L = (uint32_t)atoi(argv[ai + 1]);
        }
        if (!L) {
            L = (uint32_t)sqrt((double)n);
            if (L < 32) L = 32;
            if (L > 2000) L = 2000;
        }
        uint32_t M = L / 4 > 8 ? L / 4 : 8;

        gctx g;
        if (gctx_init(&g, k, n, B, 0)) return 1;
        gctx *gp = &g;
        size_t words = (size_t)B * g.W, wb = words * 8;
        printf("%llu*2^%llu-1: GPU NTT length %u (4 primes), %u slots, Gerbicz block %u x %u\n",
               (unsigned long long)k, (unsigned long long)n, g.NP, B, L, M);

        // x_0 = 3^k mod N on the CPU, then Montgomery form
        ahn_ctx *ctx = ahn_ctx_new(k, n, -1, NULL);
        size_t nl = ahn_limbs(ctx), nl2 = nl + 2;
        uint64_t *Nl = calloc(nl2, 8), *la = calloc(nl2, 8), *lb = calloc(nl2, 8);
        g_make_n(Nl, nl2, k, n);
        ahn_num *cx = ahn_num_new(ctx);
        ahn_set_u64(ctx, cx, 1);
        for (int b = 63; b >= 0; b--) {
            ahn_square(ctx, cx);
            if ((k >> b) & 1) ahn_mul_small(ctx, cx, 3);
        }
        memset(la, 0, nl2 * 8);
        ahn_get_limbs(ctx, cx, la);
        g_to_mont(la, nl2, k, n, Nl);
        id<MTLBuffer> bX = gctx_numbuf(&g), bD = gctx_numbuf(&g), bT = gctx_numbuf(&g),
                      bX0 = gctx_numbuf(&g);
        uint64_t *X = bX.contents, *Dn = bD.contents, *T = bT.contents, *X0 = bX0.contents;
        for (uint32_t t = 0; t < B; t++) g_limbs_to_words(la, nl2, X0 + (size_t)t * g.W, g.W);
        uint64_t *vx = malloc(wb), *vd = malloc(wb), vi = 0;
        memcpy(vx, X0, wb);
        memcpy(vd, X0, wb);
        ckhdr hdr = {{'A', 'H', 'N', 'G', 'P', 'U', '0', '1'}, k, n, B, g.W, 0};
        if (ckpath && !ck_read(ckpath, &hdr, vx, vd, words)) {
            vi = hdr.vi;
            printf("resumed from %s at iteration %llu\n", ckpath, (unsigned long long)vi);
        }

        // slot-by-slot equality of two GPU numbers after reduction mod N
        int (^equal)(const uint64_t *, const uint64_t *) = ^(const uint64_t *a, const uint64_t *b) {
            for (uint32_t t = 0; t < B; t++) {
                g_words_to_limbs(a + (size_t)t * g.W, g.W, la, nl2);
                g_words_to_limbs(b + (size_t)t * g.W, g.W, lb, nl2);
                g_reduce_mod_n(la, nl2, k, n, Nl);
                g_reduce_mod_n(lb, nl2, k, n, Nl);
                if (memcmp(la, lb, nl2 * 8)) return 0;
            }
            return 1;
        };

        double t0 = now_s(), last_ck = t0, last_pr = t0;
        uint64_t full = n / L * L, errors = 0;
        int consecutive = 0, rc = 0;
        for (;;) {
            if (vi >= full) break;
            memcpy(X, vx, wb);
            memcpy(Dn, vd, wb);
            uint64_t i = vi, check_at = vi + (uint64_t)L * M;
            if (check_at > full) check_at = full;
            while (i < check_at) {
                if (gctx_run(&g, ^(id<MTLComputeCommandEncoder> e) {
                        for (uint32_t j = 0; j < L; j++) gctx_sqr(gp, e, bX);
                    }))
                    return 1;
                i += L;
                if (fault_at && i >= fault_at) {
                    X[0] += 1;  // a valid representation of a wrong value, slot 0
                    fault_at = 0;
                    printf("injected a fault at iteration %llu\n", (unsigned long long)i);
                }
                if (i == check_at) memcpy(T, Dn, wb);
                if (gctx_run(&g, ^(id<MTLComputeCommandEncoder> e) {
                        gctx_mul(gp, e, bD, bX);
                    }))
                    return 1;
                if (now_s() - last_pr > 30) {
                    double el = now_s() - t0;
                    printf("  %5.1f%%  %.0f s elapsed\n", 100.0 * i / n, el);
                    fflush(stdout);
                    last_pr = now_s();
                }
            }
            // verify d == x0 * t^(2^L)
            if (gctx_run(&g, ^(id<MTLComputeCommandEncoder> e) {
                    for (uint32_t j = 0; j < L; j++) gctx_sqr(gp, e, bT);
                    gctx_mul(gp, e, bT, bX0);
                }))
                return 1;
            if (!equal(T, Dn)) {
                errors++;
                printf("Gerbicz check failed at iteration %llu; rolling back to %llu\n",
                       (unsigned long long)i, (unsigned long long)vi);
                if (++consecutive > 5) {
                    rc = 3;
                    break;
                }
                continue;
            }
            consecutive = 0;
            memcpy(vx, X, wb);
            memcpy(vd, Dn, wb);
            vi = i;
            if (ckpath && (now_s() - last_ck > ck_secs || vi == full)) {
                hdr.vi = vi;
                if (ck_write(ckpath, hdr, vx, vd, words)) fprintf(stderr, "checkpoint write failed\n");
                last_ck = now_s();
            }
        }
        if (rc) {
            fprintf(stderr, "too many consecutive Gerbicz failures\n");
            return rc;
        }
        // unprotected tail: twice, compared
        uint64_t tail = n - full;
        for (int attempt = 0;; attempt++) {
            memcpy(X, vx, wb);
            memcpy(T, vx, wb);
            if (gctx_run(&g, ^(id<MTLComputeCommandEncoder> e) {
                    for (uint64_t j = 0; j < tail; j++) {
                        gctx_sqr(gp, e, bX);
                        gctx_sqr(gp, e, bT);
                    }
                }))
                return 1;
            if (equal(X, T)) break;
            errors++;
            if (attempt > 5) {
                fprintf(stderr, "tail mismatch\n");
                return 3;
            }
        }
        double secs = now_s() - t0;

        // x_n = X * 2^-n = X * k (mod N); PRP iff x_n == 9; residue = x_n / 9
        uint64_t res0 = 0;
        int all_same = 1, is_prp = 0;
        for (uint32_t t = 0; t < B; t++) {
            memset(la, 0, nl2 * 8);
            g_words_to_limbs(X + (size_t)t * g.W, g.W, la, nl2);
            g_reduce_mod_n(la, nl2, k, n, Nl);
            ahn_set_limbs(ctx, cx, la);
            ahn_mul_small(ctx, cx, (int64_t)k);
            memset(la, 0, nl2 * 8);
            ahn_get_limbs(ctx, cx, la);
            int prp = la[0] == 9;
            for (size_t j = 1; j < nl; j++) prp &= la[j] == 0;
            if (ahn_residue_nm1(la, Nl, nl, 3)) return 4;
            if (t == 0) {
                res0 = la[0];
                is_prp = prp;
            } else if (la[0] != res0) {
                all_same = 0;
            }
        }
        printf("%llu*2^%llu-1 is %s  RES64 %016llX  (%s slots agree)  %.1f s for %u tests: "
               "%.1f s per test, %.0f tests/hour, Gerbicz errors %llu\n",
               (unsigned long long)k, (unsigned long long)n, is_prp ? "a probable prime!" : "composite",
               (unsigned long long)res0, all_same ? "all" : "NOT all", secs, B, secs / B,
               3600.0 * B / secs, (unsigned long long)errors);
        if (expect) {
            int match = strtoull(expect, 0, 16) == res0 && all_same;
            printf("%s expected %s\n", match ? "MATCH" : "MISMATCH", expect);
            return match ? 0 : 5;
        }
        return all_same ? 0 : 5;
    }
}
