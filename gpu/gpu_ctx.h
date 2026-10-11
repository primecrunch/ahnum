// Shared host code for the GPU squaring prototype (gpu_sq.m, gpu_prp.m):
// device and pipeline setup, per-prime NTT tables, and encoders for a squaring
// and a general product mod N = k * 2^n - 1 of B numbers at once. Numbers are
// W words of 45 bits in Montgomery form with R = 2^n; see sq.metal.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t GPR[4] = {2013265921u, 1811939329u, 469762049u, 754974721u};
static const uint32_t GGEN[4] = {31, 13, 3, 11};  // primitive roots

static uint32_t g_mulmod(uint32_t a, uint32_t b, uint32_t p) { return (uint32_t)((uint64_t)a * b % p); }
static uint32_t g_powmod(uint32_t a, uint64_t e, uint32_t p) {
    uint32_t r = 1;
    for (; e; e >>= 1, a = g_mulmod(a, a, p))
        if (e & 1) r = g_mulmod(r, a, p);
    return r;
}
static uint32_t g_invmod(uint32_t a, uint32_t p) { return g_powmod(a % p, p - 2, p); }
static uint32_t g_mont(uint32_t a, uint32_t p) { return (uint32_t)(((uint64_t)a << 32) % p); }
static uint32_t g_pinv(uint32_t p) {
    uint32_t inv = 1;
    for (int t = 0; t < 5; t++) inv *= 2 - p * inv;
    return (uint32_t)-inv;  // -p^-1 mod 2^32
}

typedef struct {
    uint32_t p, pinv;
} GParams;

typedef struct {
    uint32_t p[4], pinv[4], c32[4], c64[4];
    uint32_t inv01, inv02, inv12, inv03, inv13, inv23;
    uint32_t P01[2], P012[3];
    uint32_t k, pad;
} GGarner;

typedef struct {
    uint64_t k, n;
    uint32_t B, W, WN0, BN0, NP, R, C, AR, AC, RT, G;
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLComputePipelineState> pcf, prs, pci, prm, pres, pcrt, pcar, pfix, prd, pfold;
    NSArray *prm_b, *twR, *itwR, *twC, *itwC, *t4, *it4;  // per prime
    id<MTLBuffer> gar, bB, res, res2, parts, D, flag, lenNP, lenW;
} gctx;

// numbers live in buffers of B * W ulongs (shared storage, CPU-visible)
static id<MTLBuffer> gctx_numbuf(gctx *g) {
    return [g->dev newBufferWithLength:(size_t)g->B * g->W * 8 options:MTLResourceStorageModeShared];
}

// Returns 0 on success. force_ripple compiles the carry test mode (sq.metal).
static int gctx_init(gctx *g, uint64_t k, uint64_t n, uint32_t B, int force_ripple) {
    *g = (gctx){0};
    g->k = k;
    g->n = n;
    g->B = B;
    g->W = (uint32_t)((n + 30 + 44) / 45);
    g->WN0 = (uint32_t)(n / 45);
    g->BN0 = (uint32_t)(n % 45);
    g->NP = 1;
    while (g->NP < 2 * g->W) g->NP <<= 1;
    if (g->NP > 131072) {
        fprintf(stderr, "n too large for the prototype (NP %u > 131072)\n", g->NP);
        return -1;
    }
    g->C = g->NP >= 4096 ? 256 : 64;
    g->R = g->NP / g->C;
    g->AR = 1u << (__builtin_ctz(g->R) / 2);
    g->AC = 1u << (__builtin_ctz(g->C) / 2);
    g->RT = 8;
    g->G = 16;
    uint32_t R = g->R, C = g->C, NP = g->NP;

    g->dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *ntt = [NSString stringWithContentsOfFile:@"gpu/ntt_reg.metal"
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    NSString *sqs = [NSString stringWithContentsOfFile:@"gpu/sq.metal"
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!ntt || !sqs) {
        fprintf(stderr, "cannot read gpu/*.metal (run from the repo root)\n");
        return -1;
    }
    NSString *defs = [NSString
        stringWithFormat:@"#define R %u\n#define C %u\n#define G %u\n#define AR %u\n#define BR %u\n"
                          "#define AC %u\n#define BC %u\n#define RT %u\n#define SHOUP 1\n"
                          "#define W %u\n#define NP %u\n#define WN0 %u\n#define BN0 %u\n%s",
                         R, C, g->G, g->AR, R / g->AR, g->AC, C / g->AC, g->RT, g->W, NP, g->WN0,
                         g->BN0, force_ripple ? "#define FORCE_RIPPLE 1\n" : ""];
    NSString *src = [[defs stringByAppendingString:ntt] stringByAppendingString:sqs];
    id<MTLLibrary> lib = [g->dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String);
        return -1;
    }
    id<MTLDevice> dev = g->dev;
    id<MTLComputePipelineState> (^pipe)(NSString *) = ^(NSString *name) {
        NSError *e = nil;
        id<MTLComputePipelineState> ps =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&e];
        if (!ps) fprintf(stderr, "pipeline %s failed\n", name.UTF8String);
        return ps;
    };
    g->pcf = pipe(@"col_fwd_r");
    g->prs = pipe(@"row_sq_r");
    g->pci = pipe(@"col_inv_r");
    g->prm = pipe(@"row_mul_r");
    g->pres = pipe(@"to_res");
    g->pcrt = pipe(@"crt_split");
    g->pcar = pipe(@"carry");
    g->pfix = pipe(@"fixup_carry");
    g->prd = pipe(@"redc");
    g->pfold = pipe(@"fold");
    if (!g->pcf || !g->prs || !g->pci || !g->prm || !g->pres || !g->pcrt || !g->pcar || !g->pfix ||
        !g->prd || !g->pfold)
        return -1;

    // per-prime NTT tables (Shoup pairs: twmul(x, (v, v')) = x * v mod p)
    NSMutableArray *aP = [NSMutableArray array], *aTwR = [NSMutableArray array],
                   *aItwR = [NSMutableArray array], *aTwC = [NSMutableArray array],
                   *aItwC = [NSMutableArray array], *aT4 = [NSMutableArray array],
                   *aIt4 = [NSMutableArray array];
    for (int i = 0; i < 4; i++) {
        uint32_t p = GPR[i];
        GParams prm = {p, g_pinv(p)};
        [aP addObject:[dev newBufferWithBytes:&prm length:sizeof prm options:MTLResourceStorageModeShared]];
        uint32_t wN = g_powmod(GGEN[i], (p - 1) / NP, p), wNi = g_invmod(wN, p);
        if (g_powmod(wN, NP / 2, p) != p - 1) {
            fprintf(stderr, "prime %u: bad root\n", p);
            return -1;
        }
        uint32_t wR = g_powmod(wN, C, p), wRi = g_powmod(wNi, C, p);
        uint32_t wC = g_powmod(wN, R, p), wCi = g_powmod(wNi, R, p);
        // inverse scale 1/NP, times 2^32 for the Montgomery pointwise product
        uint32_t scale = g_mulmod(g_invmod(NP, p), (uint32_t)((1ull << 32) % p), p);
        id<MTLBuffer> tR = [dev newBufferWithLength:R * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> itR = [dev newBufferWithLength:R * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> tC = [dev newBufferWithLength:C * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> itC = [dev newBufferWithLength:C * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> f4 = [dev newBufferWithLength:(size_t)NP * 8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> i4 = [dev newBufferWithLength:(size_t)NP * 8 options:MTLResourceStorageModeShared];
        void (^put)(id<MTLBuffer>, size_t, uint32_t) = ^(id<MTLBuffer> b, size_t j, uint32_t v) {
            uint32_t *t = b.contents;
            t[2 * j] = v;
            t[2 * j + 1] = (uint32_t)(((uint64_t)v << 32) / p);
        };
        for (uint32_t j = 0; j < R; j++) {
            put(tR, j, g_powmod(wR, j, p));
            put(itR, j, g_powmod(wRi, j, p));
        }
        for (uint32_t j = 0; j < C; j++) {
            put(tC, j, g_powmod(wC, j, p));
            put(itC, j, g_powmod(wCi, j, p));
        }
        for (uint32_t r = 0; r < R; r++) {
            uint32_t a = g_powmod(wN, r, p), ai = g_powmod(wNi, r, p), x = 1, xi = scale;
            for (uint32_t c = 0; c < C; c++) {
                put(f4, (size_t)r * C + c, x);
                put(i4, (size_t)r * C + c, xi);
                x = g_mulmod(x, a, p);
                xi = g_mulmod(xi, ai, p);
            }
        }
        [aTwR addObject:tR];
        [aItwR addObject:itR];
        [aTwC addObject:tC];
        [aItwC addObject:itC];
        [aT4 addObject:f4];
        [aIt4 addObject:i4];
    }
    g->prm_b = aP;
    g->twR = aTwR;
    g->itwR = aItwR;
    g->twC = aTwC;
    g->itwC = aItwC;
    g->t4 = aT4;
    g->it4 = aIt4;

    GGarner gr = {0};
    for (int i = 0; i < 4; i++) {
        uint32_t p = GPR[i];
        gr.p[i] = p;
        gr.pinv[i] = g_pinv(p);
        gr.c32[i] = (uint32_t)((1ull << 32) % p);
        gr.c64[i] = (uint32_t)(((unsigned __int128)1 << 64) % p);
    }
    gr.inv01 = g_mont(g_invmod(GPR[0], GPR[1]), GPR[1]);
    gr.inv02 = g_mont(g_invmod(GPR[0], GPR[2]), GPR[2]);
    gr.inv12 = g_mont(g_invmod(GPR[1], GPR[2]), GPR[2]);
    gr.inv03 = g_mont(g_invmod(GPR[0], GPR[3]), GPR[3]);
    gr.inv13 = g_mont(g_invmod(GPR[1], GPR[3]), GPR[3]);
    gr.inv23 = g_mont(g_invmod(GPR[2], GPR[3]), GPR[3]);
    uint64_t p01 = (uint64_t)GPR[0] * GPR[1];
    unsigned __int128 p012 = (unsigned __int128)p01 * GPR[2];
    gr.P01[0] = (uint32_t)p01;
    gr.P01[1] = (uint32_t)(p01 >> 32);
    gr.P012[0] = (uint32_t)p012;
    gr.P012[1] = (uint32_t)(p012 >> 32);
    gr.P012[2] = (uint32_t)(p012 >> 64);
    gr.k = (uint32_t)k;
    g->gar = [dev newBufferWithBytes:&gr length:sizeof gr options:MTLResourceStorageModeShared];
    g->bB = [dev newBufferWithBytes:&g->B length:4 options:MTLResourceStorageModeShared];
    g->res = [dev newBufferWithLength:(size_t)4 * B * NP * 4 options:MTLResourceStorageModeShared];
    g->res2 = [dev newBufferWithLength:(size_t)4 * B * NP * 4 options:MTLResourceStorageModeShared];
    g->parts = [dev newBufferWithLength:(size_t)B * NP * 3 * 8 options:MTLResourceStorageModeShared];
    g->D = [dev newBufferWithLength:(size_t)B * NP * 8 options:MTLResourceStorageModeShared];
    g->flag = [dev newBufferWithLength:(size_t)B * 4 options:MTLResourceStorageModeShared];
    memset(g->flag.contents, 0, (size_t)B * 4);
    uint32_t lnp = NP, lw = g->W;
    g->lenNP = [dev newBufferWithBytes:&lnp length:4 options:MTLResourceStorageModeShared];
    g->lenW = [dev newBufferWithBytes:&lw length:4 options:MTLResourceStorageModeShared];
    g->q = [dev newCommandQueue];
    return 0;
}

static void gctx_to_res(gctx *g, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x, id<MTLBuffer> res) {
    [e setComputePipelineState:g->pres];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:res offset:0 atIndex:1];
    [e setBuffer:g->gar offset:0 atIndex:2];
    [e setBuffer:g->bB offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)g->B * g->NP, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void gctx_col(gctx *g, id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> ps,
                     id<MTLBuffer> res, int i, id<MTLBuffer> four, id<MTLBuffer> tw) {
    [e setComputePipelineState:ps];
    [e setBuffer:res offset:(size_t)i * g->B * g->NP * 4 atIndex:0];
    [e setBuffer:four offset:0 atIndex:1];
    [e setBuffer:g->prm_b[i] offset:0 atIndex:2];
    [e setBuffer:tw offset:0 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(g->B * (g->C / g->G), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(g->G * g->AR, 1, 1)];
}

// CRT, carries, reduction and fold into dst (res holds the 4 product residues)
static void gctx_finish(gctx *g, id<MTLComputeCommandEncoder> e, id<MTLBuffer> dst) {
    MTLSize t256 = MTLSizeMake(256, 1, 1), tB = MTLSizeMake(g->B < 64 ? g->B : 64, 1, 1);
    [e setComputePipelineState:g->pcrt];
    [e setBuffer:g->res offset:0 atIndex:0];
    [e setBuffer:g->parts offset:0 atIndex:1];
    [e setBuffer:g->gar offset:0 atIndex:2];
    [e setBuffer:g->bB offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)g->B * g->NP, 1, 1) threadsPerThreadgroup:t256];
    [e setComputePipelineState:g->pcar];
    [e setBuffer:g->parts offset:0 atIndex:0];
    [e setBuffer:g->D offset:0 atIndex:1];
    [e setBuffer:g->flag offset:0 atIndex:4];
    [e dispatchThreads:MTLSizeMake((size_t)g->B * g->NP, 1, 1) threadsPerThreadgroup:t256];
    [e setComputePipelineState:g->pfix];
    [e setBuffer:g->D offset:0 atIndex:1];
    [e setBuffer:g->lenNP offset:0 atIndex:6];
    [e dispatchThreads:MTLSizeMake(g->B, 1, 1) threadsPerThreadgroup:tB];
    [e setComputePipelineState:g->prd];
    [e setBuffer:g->D offset:0 atIndex:0];
    [e setBuffer:dst offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)g->B * g->W, 1, 1) threadsPerThreadgroup:t256];
    [e setComputePipelineState:g->pfix];
    [e setBuffer:g->lenW offset:0 atIndex:6];
    [e dispatchThreads:MTLSizeMake(g->B, 1, 1) threadsPerThreadgroup:tB];
    [e setComputePipelineState:g->pfold];
    [e dispatchThreads:MTLSizeMake(g->B, 1, 1) threadsPerThreadgroup:tB];
}

// x = x^2 * 2^-n mod N (Montgomery square), all B numbers
static void gctx_sqr(gctx *g, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x) {
    gctx_to_res(g, e, x, g->res);
    for (int i = 0; i < 4; i++) {
        gctx_col(g, e, g->pcf, g->res, i, g->t4[i], g->twR[i]);
        [e setComputePipelineState:g->prs];
        [e setBuffer:g->twC[i] offset:0 atIndex:3];
        [e setBuffer:g->itwC[i] offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(g->B * g->R / g->RT, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(g->RT * g->AC, 1, 1)];
        gctx_col(g, e, g->pci, g->res, i, g->it4[i], g->itwR[i]);
    }
    gctx_finish(g, e, x);
}

// dst = dst * src * 2^-n mod N (Montgomery product), all B numbers
static void gctx_mul(gctx *g, id<MTLComputeCommandEncoder> e, id<MTLBuffer> dst, id<MTLBuffer> src) {
    gctx_to_res(g, e, dst, g->res);
    gctx_to_res(g, e, src, g->res2);
    for (int i = 0; i < 4; i++) {
        gctx_col(g, e, g->pcf, g->res, i, g->t4[i], g->twR[i]);
        gctx_col(g, e, g->pcf, g->res2, i, g->t4[i], g->twR[i]);
        [e setComputePipelineState:g->prm];
        [e setBuffer:g->res offset:(size_t)i * g->B * g->NP * 4 atIndex:0];
        [e setBuffer:g->res2 offset:(size_t)i * g->B * g->NP * 4 atIndex:5];
        [e setBuffer:g->prm_b[i] offset:0 atIndex:2];
        [e setBuffer:g->twC[i] offset:0 atIndex:3];
        [e setBuffer:g->itwC[i] offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(g->B * g->R / g->RT, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(g->RT * g->AC, 1, 1)];
        gctx_col(g, e, g->pci, g->res, i, g->it4[i], g->itwR[i]);
    }
    gctx_finish(g, e, dst);
}

// Run `count` squarings of x (or, with src, one product) and wait. Returns 0, or
// -1 on a GPU error.
static int gctx_run(gctx *g, void (^body)(id<MTLComputeCommandEncoder>)) {
    id<MTLCommandBuffer> cb = [g->q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    body(e);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String);
        return -1;
    }
    return 0;
}

// ---- host-side big integers (64-bit limbs) -----------------------------------

// value of W 45-bit words -> nl limbs (zero-extended)
static void g_words_to_limbs(const uint64_t *w, uint32_t W, uint64_t *l, size_t nl) {
    memset(l, 0, nl * 8);
    for (uint32_t j = 0; j < W; j++) {
        uint64_t bit = 45ull * j, v = w[j];
        size_t li = bit / 64;
        unsigned sh = bit % 64;
        if (li < nl) l[li] |= v << sh;
        if (sh > 19 && li + 1 < nl) l[li + 1] |= v >> (64 - sh);
    }
}

static void g_limbs_to_words(const uint64_t *l, size_t nl, uint64_t *w, uint32_t W) {
    for (uint32_t j = 0; j < W; j++) {
        uint64_t bit = 45ull * j, v = 0;
        size_t li = bit / 64;
        unsigned sh = bit % 64;
        if (li < nl) v = l[li] >> sh;
        if (sh > 19 && li + 1 < nl) v |= l[li + 1] << (64 - sh);
        w[j] = v & ((1ull << 45) - 1);
    }
}

// N = k 2^n - 1 into nl limbs
static void g_make_n(uint64_t *Nl, size_t nl, uint64_t k, uint64_t n) {
    memset(Nl, 0, nl * 8);
    size_t ln = n / 64;
    unsigned __int128 kk = (unsigned __int128)k << (n % 64);
    Nl[ln] = (uint64_t)kk;
    if (ln + 1 < nl) Nl[ln + 1] = (uint64_t)(kk >> 64);
    for (size_t i = 0; i < nl; i++) {
        if (Nl[i]--) break;
    }
}

// x (nl limbs, x < 2^(n + 64)) -> x mod N: T = x >> n = q k + r gives
// x = x mod 2^n + q + r 2^n (k 2^n = 1 mod N), then one conditional subtract
static void g_reduce_mod_n(uint64_t *x, size_t nl, uint64_t k, uint64_t n, const uint64_t *N) {
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

// x < N (nl limbs) -> Montgomery form x 2^n mod N: x = q k + r (long division
// by k), x 2^n = q k 2^n + r 2^n = q + r 2^n (mod N), then reduce.
static void g_to_mont(uint64_t *x, size_t nl, uint64_t k, uint64_t n, const uint64_t *N) {
    unsigned __int128 rem = 0;
    for (size_t i = nl; i-- > 0;) {
        unsigned __int128 cur = (rem << 64) | x[i];
        x[i] = (uint64_t)(cur / k);
        rem = cur % k;
    }
    // x = q; add r 2^n
    size_t ln = n / 64;
    unsigned __int128 rr = (unsigned __int128)(uint64_t)rem << (n % 64);
    unsigned __int128 c = (uint64_t)rr;
    for (size_t i = ln; i < nl; i++) {
        c += x[i];
        if (i == ln + 1) c += (uint64_t)(rr >> 64);
        x[i] = (uint64_t)c;
        c >>= 64;
        if (!c && i > ln) break;
    }
    g_reduce_mod_n(x, nl, k, n, N);
}
