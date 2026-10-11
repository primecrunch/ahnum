// Prototype host for gpu/ntt.metal: batched NTT squaring mod p = 15*2^27 + 1
// on the GPU. Checks one squaring against a direct cyclic square on the CPU,
// then times many squarings of B independent numbers.
//
// build: clang -O2 -fobjc-arc gpu/ntt_proto.m -framework Metal -framework Foundation \
//            -o build/ntt_proto
// usage: build/ntt_proto [R C] [B] [iters]   (N = R*C, powers of two)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const uint32_t P = 2013265921u;  // 15 * 2^27 + 1, primitive root 31

static uint32_t mulmod(uint32_t a, uint32_t b) { return (uint32_t)((uint64_t)a * b % P); }
static uint32_t powmod(uint32_t a, uint64_t e) {
    uint32_t r = 1;
    for (; e; e >>= 1, a = mulmod(a, a))
        if (e & 1) r = mulmod(r, a);
    return r;
}
static uint32_t to_mont(uint32_t a) { return (uint32_t)(((uint64_t)a << 32) % P); }
static uint32_t bitrev(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; i++) r |= ((x >> i) & 1) << (bits - 1 - i);
    return r;
}

typedef struct {
    uint32_t p, pinv;
} Params;

int main(int argc, char **argv) {
    @autoreleasepool {
        uint32_t R = argc > 2 ? (uint32_t)atoi(argv[1]) : 128;
        uint32_t C = argc > 2 ? (uint32_t)atoi(argv[2]) : 256;
        uint32_t B = argc > 3 ? (uint32_t)atoi(argv[3]) : 64;
        uint32_t iters = argc > 4 ? (uint32_t)atoi(argv[4]) : 50;
        uint32_t N = R * C, G = 16, TPG_COL = 256, TPG_ROW = C / 2 < 256 ? C / 2 : 256;
        uint32_t rbits = __builtin_ctz(R);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"gpu/ntt.metal"
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        if (!src) {
            fprintf(stderr, "cannot read gpu/ntt.metal (run from the repo root)\n");
            return 1;
        }
        NSString *defs = [NSString stringWithFormat:@"#define R %u\n#define C %u\n#define G %u\n"
                                                     "#define TPG_COL %u\n#define TPG_ROW %u\n",
                                                    R, C, G, TPG_COL, TPG_ROW];
        id<MTLLibrary> lib = [dev newLibraryWithSource:[defs stringByAppendingString:src]
                                               options:nil
                                                 error:&err];
        if (!lib) {
            fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> pcf =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"col_fwd"] error:&err];
        id<MTLComputePipelineState> prs =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"row_sq"] error:&err];
        id<MTLComputePipelineState> pci =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"col_inv"] error:&err];
        if (!pcf || !prs || !pci) {
            fprintf(stderr, "pipeline failed: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }

        // Montgomery constant -p^-1 mod 2^32 by Newton iteration
        uint32_t inv = 1;
        for (int i = 0; i < 5; i++) inv *= 2 - P * inv;
        Params prm = {P, (uint32_t)-inv};

        uint32_t wN = powmod(31, (P - 1) / N), wNi = powmod(wN, P - 2);
        uint32_t wR = powmod(wN, C), wRi = powmod(wNi, C);  // w_R = w_N^C
        uint32_t wC = powmod(wN, R), wCi = powmod(wNi, R);  // w_C = w_N^R
        uint32_t Ninv = powmod(N, P - 2);
        // the pointwise mmul(x, x) leaves a 2^-32 factor: fold 2^32 into the scale
        uint32_t scale = mulmod(Ninv, (uint32_t)(((uint64_t)1 << 32) % P));

        id<MTLBuffer> bTwR = [dev newBufferWithLength:R / 2 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bItwR = [dev newBufferWithLength:R / 2 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bTwC = [dev newBufferWithLength:C / 2 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bItwC = [dev newBufferWithLength:C / 2 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bT4 = [dev newBufferWithLength:(size_t)N * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bIt4 = [dev newBufferWithLength:(size_t)N * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bData = [dev newBufferWithLength:(size_t)B * N * 4
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bPrm = [dev newBufferWithBytes:&prm length:sizeof prm
                                             options:MTLResourceStorageModeShared];
        uint32_t *twR = bTwR.contents, *itwR = bItwR.contents, *twC = bTwC.contents,
                 *itwC = bItwC.contents, *t4 = bT4.contents, *it4 = bIt4.contents;
        for (uint32_t i = 0; i < R / 2; i++) {
            twR[i] = to_mont(powmod(wR, i));
            itwR[i] = to_mont(powmod(wRi, i));
        }
        for (uint32_t i = 0; i < C / 2; i++) {
            twC[i] = to_mont(powmod(wC, i));
            itwC[i] = to_mont(powmod(wCi, i));
        }
        for (uint32_t r = 0; r < R; r++) {
            uint32_t k1 = bitrev(r, rbits);  // stored row r holds frequency k1
            for (uint32_t c = 0; c < C; c++) {
                t4[r * C + c] = to_mont(powmod(wN, (uint64_t)k1 * c));
                it4[r * C + c] = to_mont(mulmod(powmod(wNi, (uint64_t)k1 * c), scale));
            }
        }

        uint32_t *data = bData.contents;
        srand(1);
        for (size_t i = 0; i < (size_t)B * N; i++) data[i] = (uint32_t)(((uint64_t)rand() << 16 ^ rand()) % P);
        uint32_t *orig = malloc((size_t)N * 4);
        memcpy(orig, data, (size_t)N * 4);

        id<MTLCommandQueue> q = [dev newCommandQueue];
        void (^encode)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pcf];
            [e setBuffer:bData offset:0 atIndex:0];
            [e setBuffer:bTwR offset:0 atIndex:1];
            [e setBuffer:bT4 offset:0 atIndex:2];
            [e setBuffer:bPrm offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(B * (C / G), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(TPG_COL, 1, 1)];
            [e setComputePipelineState:prs];
            [e setBuffer:bTwC offset:0 atIndex:1];
            [e setBuffer:bItwC offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(B * R, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(TPG_ROW, 1, 1)];
            [e setComputePipelineState:pci];
            [e setBuffer:bItwR offset:0 atIndex:1];
            [e setBuffer:bIt4 offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(B * (C / G), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(TPG_COL, 1, 1)];
        };

        // correctness: one squaring, compare test 0 at 64 positions
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            encode(e);
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            int bad = 0;
            for (int t = 0; t < 64; t++) {
                uint32_t k = (uint32_t)((uint64_t)t * 2654435761u % N);
                uint64_t acc = 0;
                for (uint32_t i = 0; i < N; i++) {
                    acc += (uint64_t)orig[i] * orig[(k + N - i) % N] % P;
                    if (acc >= (1ull << 62)) acc %= P;
                }
                if (acc % P != data[k]) bad++;
            }
            printf("N=%u (R=%u C=%u): check %s\n", N, R, C, bad ? "FAILED" : "ok (64 positions)");
            if (bad) return 1;
        }

        // timing: iters squarings of all B numbers in one command buffer;
        // AHN_GPU_SECS=S keeps going for S seconds and reports the average
        // (for runs alongside CPU load)
        const char *secs_env = getenv("AHN_GPU_SECS");
        if (secs_env) {
            double secs = atof(secs_env), busy = 0;
            uint64_t sq = 0;
            NSDate *t0 = [NSDate date];
            while (-[t0 timeIntervalSinceNow] < secs) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                for (uint32_t it = 0; it < iters; it++) encode(e);
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                busy += cb.GPUEndTime - cb.GPUStartTime;
                sq += (uint64_t)iters * B;
            }
            double wall = -[t0 timeIntervalSinceNow];
            printf("sustained %.1f s: %.2f us per squaring mod one prime (wall), GPU busy %.0f%%\n",
                   wall, wall / sq * 1e6, 100 * busy / wall);
            return 0;
        }
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            for (uint32_t it = 0; it < iters; it++) encode(e);
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double dt = cb.GPUEndTime - cb.GPUStartTime;
            if (dt < best) best = dt;
        }
        double per = best / iters / B * 1e6;
        printf("B=%u iters=%u: %.2f us per squaring mod one prime (%.0f squarings/s)\n", B, iters, per,
               1e6 / per);
    }
    return 0;
}
