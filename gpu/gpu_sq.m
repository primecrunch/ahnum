// Prototype: full squaring mod N = k * 2^n - 1 on the GPU, B numbers at once.
// Per squaring (all on the GPU):
//   to_res     45-bit words -> residues mod 4 NTT primes, zero-padded to NP
//   4 x NTT    squaring mod each prime (register kernels, Shoup twiddles,
//              gpu/ntt_reg.metal), three dispatches each
//   crt_split  Garner CRT -> 128-bit coefficient -> three 45-bit parts
//   carry      product words
//   redc       Montgomery reduction with R = 2^n: P * 2^-n = B + k A
//   fold       bits above n: T = q k + r -> q + r 2^n
// Numbers are in Montgomery form, so the step computes x -> x^2 * 2^-n = x^2 k
// (mod N). The check runs the same recurrence on the CPU with ahnum.
//
// build: clang -O2 -fobjc-arc -Iinclude gpu/gpu_sq.m build/libahnum.a \
//            -framework Metal -framework Foundation -o build/gpu_sq
// usage: build/gpu_sq K N [B] [squarings]   (run from the repo root)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ahnum.h"

static const uint32_t PR[4] = {2013265921u, 1811939329u, 469762049u, 754974721u};
static const uint32_t GEN[4] = {31, 13, 3, 11};  // primitive roots

static uint32_t mulmod(uint32_t a, uint32_t b, uint32_t p) { return (uint32_t)((uint64_t)a * b % p); }
static uint32_t powmod(uint32_t a, uint64_t e, uint32_t p) {
    uint32_t r = 1;
    for (; e; e >>= 1, a = mulmod(a, a, p))
        if (e & 1) r = mulmod(r, a, p);
    return r;
}
static uint32_t invmod(uint32_t a, uint32_t p) { return powmod(a % p, p - 2, p); }
static uint32_t mont(uint32_t a, uint32_t p) { return (uint32_t)(((uint64_t)a << 32) % p); }

typedef struct {
    uint32_t p, pinv;
} Params;

typedef struct {
    uint32_t p[4], pinv[4], c32[4], c64[4];
    uint32_t inv01, inv02, inv12, inv03, inv13, inv23;
    uint32_t P01[2], P012[3];
    uint32_t k, pad;
} Garner;

// value of W 45-bit words -> 64-bit limbs (nl limbs, zero-extended)
static void words_to_limbs(const uint64_t *w, uint32_t W, uint64_t *l, size_t nl) {
    memset(l, 0, nl * 8);
    for (uint32_t j = 0; j < W; j++) {
        uint64_t bit = 45ull * j, v = w[j];
        size_t li = bit / 64;
        unsigned sh = bit % 64;
        if (li < nl) l[li] |= v << sh;
        if (sh > 19 && li + 1 < nl) l[li + 1] |= v >> (64 - sh);
    }
}

// x (nl limbs, x < 2^(n + 64)) -> x mod N, N = k 2^n - 1, using the fold
// T = x >> n = q k + r: x = x mod 2^n + q + r 2^n, then one conditional subtract
static void reduce_mod_n(uint64_t *x, size_t nl, uint64_t k, uint64_t n, const uint64_t *N) {
    size_t ln = n / 64;
    unsigned bn = n % 64;
    unsigned __int128 T = 0;
    for (size_t i = ln; i < nl && i < ln + 3; i++) {
        unsigned __int128 v = x[i];
        unsigned sh = 64 * (unsigned)(i - ln) - bn;
        if (i == ln) v >>= bn;
        else if (sh < 128) v <<= sh;
        else continue;
        T |= v;
    }
    uint64_t q = (uint64_t)(T / k), r = (uint64_t)(T % k);
    if (bn) x[ln] &= (1ull << bn) - 1;
    else x[ln] = 0;
    for (size_t i = ln + 1; i < nl; i++) x[i] = 0;
    unsigned __int128 rr = (unsigned __int128)r << bn;
    x[ln] |= (uint64_t)rr;
    if (ln + 1 < nl) x[ln + 1] |= (uint64_t)(rr >> 64);
    unsigned __int128 c = q;
    for (size_t i = 0; c && i < nl; i++) {
        c += x[i];
        x[i] = (uint64_t)c;
        c >>= 64;
    }
    // x < 2N now; subtract N if x >= N
    int ge = 1;
    for (size_t i = nl; i-- > 0;) {
        if (x[i] != N[i]) {
            ge = x[i] > N[i];
            break;
        }
    }
    if (ge) {
        unsigned __int128 b = 0;
        for (size_t i = 0; i < nl; i++) {
            unsigned __int128 d = (unsigned __int128)x[i] - N[i] - b;
            x[i] = (uint64_t)d;
            b = (d >> 64) & 1;
        }
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s K N [B] [squarings]\n", argv[0]);
            return 2;
        }
        uint64_t k = strtoull(argv[1], 0, 10), n = strtoull(argv[2], 0, 10);
        uint32_t B = argc > 3 ? (uint32_t)atoi(argv[3]) : 64;
        uint32_t sq = argc > 4 ? (uint32_t)atoi(argv[4]) : 20;
        uint32_t W = (uint32_t)((n + 30 + 44) / 45), WN0 = (uint32_t)(n / 45), BN0 = (uint32_t)(n % 45);
        uint32_t NP = 1;
        while (NP < 2 * W) NP <<= 1;
        uint32_t C = NP >= 65536 ? 256 : NP >= 4096 ? 256 : 64, R = NP / C;
        uint32_t AR = 1u << (__builtin_ctz(R) / 2), BR = R / AR;
        uint32_t AC = 1u << (__builtin_ctz(C) / 2), BC = C / AC, RT = 8, G = 16;
        printf("k=%llu n=%llu: W=%u words of 45 bits, NP=%u (R=%u C=%u), B=%u\n",
               (unsigned long long)k, (unsigned long long)n, W, NP, R, C, B);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        NSString *ntt = [NSString stringWithContentsOfFile:@"gpu/ntt_reg.metal"
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        NSString *sqs = [NSString stringWithContentsOfFile:@"gpu/sq.metal"
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        if (!ntt || !sqs) {
            fprintf(stderr, "cannot read gpu/*.metal (run from the repo root)\n");
            return 1;
        }
        NSString *defs = [NSString
            stringWithFormat:@"#define R %u\n#define C %u\n#define G %u\n#define AR %u\n#define BR %u\n"
                              "#define AC %u\n#define BC %u\n#define RT %u\n#define SHOUP 1\n"
                              "#define W %u\n#define NP %u\n#define WN0 %u\n#define BN0 %u\n",
                             R, C, G, AR, BR, AC, BC, RT, W, NP, WN0, BN0];
        NSString *src = [[defs stringByAppendingString:ntt] stringByAppendingString:sqs];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> (^pipe)(NSString *) = ^(NSString *name) {
            NSError *e = nil;
            id<MTLComputePipelineState> ps =
                [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&e];
            if (!ps) fprintf(stderr, "pipeline %s failed\n", name.UTF8String);
            return ps;
        };
        id<MTLComputePipelineState> pcf = pipe(@"col_fwd_r"), prs = pipe(@"row_sq_r"),
                                    pci = pipe(@"col_inv_r"), pres = pipe(@"to_res"),
                                    pcrt = pipe(@"crt_split"), pcar = pipe(@"carry"),
                                    prd = pipe(@"redc"), pfold = pipe(@"fold");
        if (!pcf || !prs || !pci || !pres || !pcrt || !pcar || !prd || !pfold) return 1;

        // per-prime NTT tables (Shoup pairs), as in ntt_proto.m
        id<MTLBuffer> bPrm[4], bTwR[4], bItwR[4], bTwC[4], bItwC[4], bT4[4], bIt4[4];
        for (int i = 0; i < 4; i++) {
            uint32_t p = PR[i], inv = 1;
            for (int t = 0; t < 5; t++) inv *= 2 - p * inv;
            Params prm = {p, (uint32_t)-inv};
            bPrm[i] = [dev newBufferWithBytes:&prm length:sizeof prm options:MTLResourceStorageModeShared];
            uint32_t wN = powmod(GEN[i], (p - 1) / NP, p), wNi = invmod(wN, p);
            if (powmod(wN, NP / 2, p) != p - 1) {
                fprintf(stderr, "prime %u: bad root\n", p);
                return 1;
            }
            uint32_t wR = powmod(wN, C, p), wRi = powmod(wNi, C, p);
            uint32_t wC = powmod(wN, R, p), wCi = powmod(wNi, R, p);
            uint32_t scale = mulmod(invmod(NP, p), (uint32_t)((1ull << 32) % p), p);
            bTwR[i] = [dev newBufferWithLength:R * 8 options:MTLResourceStorageModeShared];
            bItwR[i] = [dev newBufferWithLength:R * 8 options:MTLResourceStorageModeShared];
            bTwC[i] = [dev newBufferWithLength:C * 8 options:MTLResourceStorageModeShared];
            bItwC[i] = [dev newBufferWithLength:C * 8 options:MTLResourceStorageModeShared];
            bT4[i] = [dev newBufferWithLength:(size_t)NP * 8 options:MTLResourceStorageModeShared];
            bIt4[i] = [dev newBufferWithLength:(size_t)NP * 8 options:MTLResourceStorageModeShared];
            void (^put)(id<MTLBuffer>, size_t, uint32_t) = ^(id<MTLBuffer> b, size_t j, uint32_t v) {
                uint32_t *t = b.contents;
                t[2 * j] = v;
                t[2 * j + 1] = (uint32_t)(((uint64_t)v << 32) / p);
            };
            for (uint32_t j = 0; j < R; j++) {
                put(bTwR[i], j, powmod(wR, j, p));
                put(bItwR[i], j, powmod(wRi, j, p));
            }
            for (uint32_t j = 0; j < C; j++) {
                put(bTwC[i], j, powmod(wC, j, p));
                put(bItwC[i], j, powmod(wCi, j, p));
            }
            for (uint32_t r = 0; r < R; r++) {
                uint32_t a = powmod(wN, r, p), ai = powmod(wNi, r, p), x = 1, xi = scale;
                for (uint32_t c = 0; c < C; c++) {
                    put(bT4[i], (size_t)r * C + c, x);
                    put(bIt4[i], (size_t)r * C + c, xi);
                    x = mulmod(x, a, p);
                    xi = mulmod(xi, ai, p);
                }
            }
        }

        Garner g = {0};
        for (int i = 0; i < 4; i++) {
            uint32_t p = PR[i], inv = 1;
            for (int t = 0; t < 5; t++) inv *= 2 - p * inv;
            g.p[i] = p;
            g.pinv[i] = (uint32_t)-inv;
            g.c32[i] = (uint32_t)((1ull << 32) % p);
            g.c64[i] = (uint32_t)(((unsigned __int128)1 << 64) % p);
        }
        g.inv01 = mont(invmod(PR[0], PR[1]), PR[1]);
        g.inv02 = mont(invmod(PR[0], PR[2]), PR[2]);
        g.inv12 = mont(invmod(PR[1], PR[2]), PR[2]);
        g.inv03 = mont(invmod(PR[0], PR[3]), PR[3]);
        g.inv13 = mont(invmod(PR[1], PR[3]), PR[3]);
        g.inv23 = mont(invmod(PR[2], PR[3]), PR[3]);
        uint64_t p01 = (uint64_t)PR[0] * PR[1];
        unsigned __int128 p012 = (unsigned __int128)p01 * PR[2];
        g.P01[0] = (uint32_t)p01;
        g.P01[1] = (uint32_t)(p01 >> 32);
        g.P012[0] = (uint32_t)p012;
        g.P012[1] = (uint32_t)(p012 >> 32);
        g.P012[2] = (uint32_t)(p012 >> 64);
        g.k = (uint32_t)k;
        id<MTLBuffer> bG = [dev newBufferWithBytes:&g length:sizeof g options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [dev newBufferWithBytes:&B length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bX = [dev newBufferWithLength:(size_t)B * W * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRes = [dev newBufferWithLength:(size_t)4 * B * NP * 4
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> bParts = [dev newBufferWithLength:(size_t)B * NP * 3 * 8
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bD = [dev newBufferWithLength:(size_t)B * NP * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOvf = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];

        // starting value: random x0 < 2^n, the same in every slot
        ahn_ctx *ctx = ahn_ctx_new(k, n, -1, NULL);
        size_t nl = ahn_limbs(ctx);
        uint64_t *x0 = calloc(nl, 8), *Nl = calloc(nl + 2, 8), *got = calloc(nl + 2, 8), *want = calloc(nl, 8);
        srand(7);
        for (size_t i = 0; i < nl; i++) x0[i] = ((uint64_t)rand() << 40) ^ ((uint64_t)rand() << 20) ^ rand();
        size_t ln = n / 64;
        if (n % 64) x0[ln] &= (1ull << (n % 64)) - 1;
        else x0[ln] = 0;
        for (size_t i = ln + 1; i < nl; i++) x0[i] = 0;
        {
            // N = k 2^n - 1
            unsigned __int128 kk = (unsigned __int128)k << (n % 64);
            Nl[ln] = (uint64_t)kk;
            if (ln + 1 < nl) Nl[ln + 1] = (uint64_t)(kk >> 64);
            unsigned __int128 b = 1;
            for (size_t i = 0; b && i < nl; i++) {
                unsigned __int128 d = (unsigned __int128)Nl[i] - b;
                Nl[i] = (uint64_t)d;
                b = (d >> 64) & 1;
            }
        }
        uint64_t *xw = bX.contents;
        for (uint32_t t = 0; t < B; t++) {
            for (uint32_t j = 0; j < W; j++) {
                uint64_t bit = 45ull * j, v = 0;
                size_t li = bit / 64;
                unsigned sh = bit % 64;
                if (li < nl) v = x0[li] >> sh;
                if (sh > 19 && li + 1 < nl) v |= x0[li + 1] << (64 - sh);
                xw[(size_t)t * W + j] = v & ((1ull << 45) - 1);
            }
        }

        // blocks cannot capture C arrays: per-prime buffers as NSArrays
        NSArray *aPrm = [NSArray arrayWithObjects:bPrm count:4], *aTwR = [NSArray arrayWithObjects:bTwR count:4],
                *aItwR = [NSArray arrayWithObjects:bItwR count:4], *aTwC = [NSArray arrayWithObjects:bTwC count:4],
                *aItwC = [NSArray arrayWithObjects:bItwC count:4], *aT4 = [NSArray arrayWithObjects:bT4 count:4],
                *aIt4 = [NSArray arrayWithObjects:bIt4 count:4];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        MTLSize tg256 = MTLSizeMake(256, 1, 1);
        void (^encode)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pres];
            [e setBuffer:bX offset:0 atIndex:0];
            [e setBuffer:bRes offset:0 atIndex:1];
            [e setBuffer:bG offset:0 atIndex:2];
            [e setBuffer:bB offset:0 atIndex:3];
            [e dispatchThreads:MTLSizeMake((size_t)B * NP, 1, 1) threadsPerThreadgroup:tg256];
            for (int i = 0; i < 4; i++) {
                size_t off = (size_t)i * B * NP * 4;
                [e setComputePipelineState:pcf];
                [e setBuffer:bRes offset:off atIndex:0];
                [e setBuffer:aT4[i] offset:0 atIndex:1];
                [e setBuffer:aPrm[i] offset:0 atIndex:2];
                [e setBuffer:aTwR[i] offset:0 atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(B * (C / G), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(G * AR, 1, 1)];
                [e setComputePipelineState:prs];
                [e setBuffer:aTwC[i] offset:0 atIndex:3];
                [e setBuffer:aItwC[i] offset:0 atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake(B * R / RT, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(RT * AC, 1, 1)];
                [e setComputePipelineState:pci];
                [e setBuffer:aIt4[i] offset:0 atIndex:1];
                [e setBuffer:aItwR[i] offset:0 atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(B * (C / G), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(G * AR, 1, 1)];
            }
            [e setComputePipelineState:pcrt];
            [e setBuffer:bRes offset:0 atIndex:0];
            [e setBuffer:bParts offset:0 atIndex:1];
            [e setBuffer:bG offset:0 atIndex:2];
            [e setBuffer:bB offset:0 atIndex:3];
            [e dispatchThreads:MTLSizeMake((size_t)B * NP, 1, 1) threadsPerThreadgroup:tg256];
            [e setComputePipelineState:pcar];
            [e setBuffer:bParts offset:0 atIndex:0];
            [e setBuffer:bD offset:0 atIndex:1];
            [e setBuffer:bOvf offset:0 atIndex:4];
            [e dispatchThreads:MTLSizeMake((size_t)B * NP, 1, 1) threadsPerThreadgroup:tg256];
            [e setComputePipelineState:prd];
            [e setBuffer:bD offset:0 atIndex:0];
            [e setBuffer:bX offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake((size_t)B * W, 1, 1) threadsPerThreadgroup:tg256];
            [e setComputePipelineState:pfold];
            [e dispatchThreads:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(B < 64 ? B : 64, 1, 1)];
        };

        // correctness: sq squarings, compare slots 0 and B-1 with the CPU
        *(uint32_t *)bOvf.contents = 0;
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            for (uint32_t s = 0; s < sq; s++) encode(e);
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String);
                return 1;
            }
        }
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
            words_to_limbs(xw + (size_t)slots[si] * W, W, got, nl + 2);
            reduce_mod_n(got, nl + 2, k, n, Nl);
            if (memcmp(got, want, nl * 8) || got[nl] || got[nl + 1]) bad++;
        }
        printf("%u squarings: %s (carry overflows: %u)\n", sq, bad ? "MISMATCH" : "match",
               *(uint32_t *)bOvf.contents);
        if (bad) return 1;

        // AHN_GPU_SECS=S: keep squaring for S seconds and report the average
        // (for runs alongside CPU load)
        const char *secs_env = getenv("AHN_GPU_SECS");
        if (secs_env) {
            double secs = atof(secs_env), busy = 0;
            uint64_t done = 0;
            NSDate *t0 = [NSDate date];
            while (-[t0 timeIntervalSinceNow] < secs) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                for (uint32_t s = 0; s < sq; s++) encode(e);
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                busy += cb.GPUEndTime - cb.GPUStartTime;
                done += (uint64_t)sq * B;
            }
            double wall = -[t0 timeIntervalSinceNow], per = wall / done * 1e6;
            printf("sustained %.1f s: %.2f us per squaring (wall), GPU busy %.0f%%, %.0f tests/hour\n",
                   wall, per, 100 * busy / wall, 3600 / (per * 1e-6 * n));
            return 0;
        }
        // timing
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            for (uint32_t s = 0; s < sq; s++) encode(e);
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double dt = cb.GPUEndTime - cb.GPUStartTime;
            if (dt < best) best = dt;
        }
        double per = best / sq / B * 1e6;
        printf("%.2f us per squaring (GPU throughput, %u in flight): %.0f s per test of %llu squarings\n",
               per, B, per * 1e-6 * n, (unsigned long long)n);
    }
    return 0;
}
