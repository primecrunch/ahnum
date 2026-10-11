// Prototype: batched NTT squaring mod one 31-bit prime on the GPU, four-step
// N = R x C, mirroring the CPU two-pass structure:
//   col_fwd: per group of G columns, DIF NTT of length R in threadgroup memory,
//            then the four-step twiddle w_N^{k1 * n2}
//   row_sq:  per row, DIF NTT of length C, pointwise square, DIT inverse
//   col_inv: inverse four-step twiddle (with the 1/N scale), DIT of length R
// The host prepends #defines for R, C, G, TPG_COL, TPG_ROW.
// Arithmetic is Montgomery with 2^32: twiddles are stored as w * 2^32 mod p,
// so mmul(x, w') = x * w mod p. Data stays in the normal domain except that the
// pointwise square mmul(x, x) leaves a 2^-32 factor, which the inverse table
// cancels.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint p, pinv;  // pinv = -p^-1 mod 2^32
};

static inline uint mmul(uint a, uint b, uint p, uint pinv) {
    uint lo = a * b, hi = mulhi(a, b);
    uint m = lo * pinv;
    uint r = hi + mulhi(m, p) + (lo != 0 ? 1u : 0u);
    return r >= p ? r - p : r;
}
static inline uint madd(uint a, uint b, uint p) {
    uint r = a + b;
    return r >= p ? r - p : r;
}
static inline uint msub(uint a, uint b, uint p) {
    return a >= b ? a - b : a + p - b;
}

// DIF over L points held at s[i * str + off]; tw[i] = w_L^i (i < L/2);
// natural order in, bit-reversed out. Threads tid < nthr share the work.
static inline void dif(threadgroup uint *s, uint L, uint str, uint off, device const uint *tw,
                       uint tid, uint nthr, constant Params &P) {
    for (uint h = L / 2; h >= 1; h >>= 1) {
        for (uint b = tid; b < L / 2; b += nthr) {
            uint blk = b / h, j = b - blk * h;
            uint i0 = (blk * 2 * h + j) * str + off, i1 = i0 + h * str;
            uint x = s[i0], y = s[i1];
            s[i0] = madd(x, y, P.p);
            s[i1] = mmul(msub(x, y, P.p), tw[j * (L / (2 * h))], P.p, P.pinv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// DIT inverse: bit-reversed in, natural out; itw[i] = w_L^-i.
static inline void dit(threadgroup uint *s, uint L, uint str, uint off, device const uint *itw,
                       uint tid, uint nthr, constant Params &P) {
    for (uint h = 1; h <= L / 2; h <<= 1) {
        for (uint b = tid; b < L / 2; b += nthr) {
            uint blk = b / h, j = b - blk * h;
            uint i0 = (blk * 2 * h + j) * str + off, i1 = i0 + h * str;
            uint x = s[i0], y = mmul(s[i1], itw[j * (L / (2 * h))], P.p, P.pinv);
            s[i0] = madd(x, y, P.p);
            s[i1] = msub(x, y, P.p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One threadgroup per (test, group of G columns). G columns of R rows sit in
// threadgroup memory as s[r * G + c].
kernel void col_fwd(device uint *data [[buffer(0)]], device const uint *twR [[buffer(1)]],
                    device const uint *t4 [[buffer(2)]], constant Params &P [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint s[R * G];
    uint test = tg / (C / G), cg = tg % (C / G);
    device uint *d = data + (size_t)test * R * C + cg * G;
    for (uint i = tid; i < R * G; i += TPG_COL) s[i] = d[(i / G) * C + i % G];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // G interleaved length-R transforms; the butterflies of all G columns are
    // spread over the threads
    for (uint h = R / 2; h >= 1; h >>= 1) {
        for (uint b = tid; b < G * R / 2; b += TPG_COL) {
            uint c = b % G, bb = b / G;
            uint blk = bb / h, j = bb - blk * h;
            uint i0 = (blk * 2 * h + j) * G + c, i1 = i0 + h * G;
            uint x = s[i0], y = s[i1];
            s[i0] = madd(x, y, P.p);
            s[i1] = mmul(msub(x, y, P.p), twR[j * (R / (2 * h))], P.p, P.pinv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < R * G; i += TPG_COL) {
        uint r = i / G, c = i % G;
        d[r * C + c] = mmul(s[i], t4[r * C + cg * G + c], P.p, P.pinv);
    }
}

kernel void row_sq(device uint *data [[buffer(0)]], device const uint *twC [[buffer(1)]],
                   device const uint *itwC [[buffer(2)]], constant Params &P [[buffer(3)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint s[C];
    device uint *d = data + (size_t)tg * C;  // rows of all tests are contiguous
    for (uint i = tid; i < C; i += TPG_ROW) s[i] = d[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dif(s, C, 1, 0, twC, tid, TPG_ROW, P);
    for (uint i = tid; i < C; i += TPG_ROW) s[i] = mmul(s[i], s[i], P.p, P.pinv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dit(s, C, 1, 0, itwC, tid, TPG_ROW, P);
    for (uint i = tid; i < C; i += TPG_ROW) d[i] = s[i];
}

kernel void col_inv(device uint *data [[buffer(0)]], device const uint *itwR [[buffer(1)]],
                    device const uint *it4 [[buffer(2)]], constant Params &P [[buffer(3)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint s[R * G];
    uint test = tg / (C / G), cg = tg % (C / G);
    device uint *d = data + (size_t)test * R * C + cg * G;
    for (uint i = tid; i < R * G; i += TPG_COL) {
        uint r = i / G, c = i % G;
        s[i] = mmul(d[r * C + c], it4[r * C + cg * G + c], P.p, P.pinv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h <= R / 2; h <<= 1) {
        for (uint b = tid; b < G * R / 2; b += TPG_COL) {
            uint c = b % G, bb = b / G;
            uint blk = bb / h, j = bb - blk * h;
            uint i0 = (blk * 2 * h + j) * G + c, i1 = i0 + h * G;
            uint x = s[i0], y = mmul(s[i1], itwR[j * (R / (2 * h))], P.p, P.pinv);
            s[i0] = madd(x, y, P.p);
            s[i1] = msub(x, y, P.p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < R * G; i += TPG_COL) d[(i / G) * C + i % G] = s[i];
}
