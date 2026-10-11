// Prototype benchmark: full squarings mod N = k * 2^n - 1 on the GPU, B numbers
// at once (see gpu_ctx.h / sq.metal for the steps). Numbers are in Montgomery
// form, so each step is x -> x^2 * 2^-n = x^2 k (mod N); the check runs the
// same recurrence on the CPU with ahnum.
//
// build: clang -O2 -fobjc-arc -Iinclude gpu/gpu_sq.m build/libahnum.a \
//            -framework Metal -framework Foundation -o build/gpu_sq
// usage: build/gpu_sq K N [B] [squarings]   (run from the repo root)
//   AHN_GPU_SECS=S          keep squaring for S seconds (runs alongside CPU load)
//   AHN_GPU_FORCE_RIPPLE=1  test mode: every carry goes through the sequential fix-up
#include "gpu_ctx.h"

#include "ahnum.h"

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s K N [B] [squarings]\n", argv[0]);
            return 2;
        }
        uint64_t k = strtoull(argv[1], 0, 10), n = strtoull(argv[2], 0, 10);
        uint32_t B = argc > 3 ? (uint32_t)atoi(argv[3]) : 64;
        uint32_t sq = argc > 4 ? (uint32_t)atoi(argv[4]) : 20;
        int force = getenv("AHN_GPU_FORCE_RIPPLE") != NULL;
        gctx g;
        if (gctx_init(&g, k, n, B, force)) return 1;
        printf("k=%llu n=%llu: W=%u words of 45 bits, NP=%u (R=%u C=%u), B=%u%s\n",
               (unsigned long long)k, (unsigned long long)n, g.W, g.NP, g.R, g.C, B,
               force ? ", forced ripple" : "");

        ahn_ctx *ctx = ahn_ctx_new(k, n, -1, NULL);
        size_t nl = ahn_limbs(ctx);
        uint64_t *x0 = calloc(nl, 8), *Nl = calloc(nl + 2, 8), *got = calloc(nl + 2, 8),
                 *want = calloc(nl, 8);
        g_make_n(Nl, nl + 2, k, n);
        srand(7);
        for (size_t i = 0; i < nl; i++) x0[i] = ((uint64_t)rand() << 40) ^ ((uint64_t)rand() << 20) ^ rand();
        size_t ln = n / 64;  // x0 < 2^n
        if (n % 64) x0[ln] &= (1ull << (n % 64)) - 1;
        else x0[ln] = 0;
        for (size_t i = ln + 1; i < nl; i++) x0[i] = 0;
        id<MTLBuffer> bX = gctx_numbuf(&g);
        uint64_t *xw = bX.contents;
        for (uint32_t t = 0; t < B; t++) g_limbs_to_words(x0, nl, xw + (size_t)t * g.W, g.W);

        gctx *gp = &g;
        if (gctx_run(&g, ^(id<MTLComputeCommandEncoder> e) {
                for (uint32_t s = 0; s < sq; s++) gctx_sqr(gp, e, bX);
            }))
            return 1;
        ahn_num *cx = ahn_num_new(ctx);
        ahn_set_limbs(ctx, cx, x0);
        for (uint32_t s = 0; s < sq; s++) {
            ahn_square(ctx, cx);
            ahn_mul_small(ctx, cx, (int64_t)k);
        }
        ahn_get_limbs(ctx, cx, want);
        int bad = 0;
        uint32_t slots[2] = {0, B - 1};
        for (int si = 0; si < 2; si++) {
            g_words_to_limbs(xw + (size_t)slots[si] * g.W, g.W, got, nl + 2);
            g_reduce_mod_n(got, nl + 2, k, n, Nl);
            if (memcmp(got, want, nl * 8) || got[nl] || got[nl + 1]) bad++;
        }
        printf("%u squarings: %s\n", sq, bad ? "MISMATCH" : "match");
        if (bad) return 1;

        const char *secs_env = getenv("AHN_GPU_SECS");
        if (secs_env) {
            double secs = atof(secs_env);
            uint64_t done = 0;
            NSDate *t0 = [NSDate date];
            while (-[t0 timeIntervalSinceNow] < secs) {
                gctx_run(&g, ^(id<MTLComputeCommandEncoder> e) {
                    for (uint32_t s = 0; s < sq; s++) gctx_sqr(gp, e, bX);
                });
                done += (uint64_t)sq * B;
            }
            double wall = -[t0 timeIntervalSinceNow], per = wall / done * 1e6;
            printf("sustained %.1f s: %.2f us per squaring (wall), %.0f tests/hour\n", wall, per,
                   3600 / (per * 1e-6 * n));
            return 0;
        }
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            NSDate *t0 = [NSDate date];
            gctx_run(&g, ^(id<MTLComputeCommandEncoder> e) {
                for (uint32_t s = 0; s < sq; s++) gctx_sqr(gp, e, bX);
            });
            double dt = -[t0 timeIntervalSinceNow];
            if (dt < best) best = dt;
        }
        double per = best / sq / B * 1e6;
        printf("%.2f us per squaring (%u in flight): %.0f s per test of %llu squarings\n", per, B,
               per * 1e-6 * n, (unsigned long long)n);
    }
    return 0;
}
