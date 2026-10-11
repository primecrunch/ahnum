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

// Stage loops over ncol interleaved columns of L points, s[i * ncol + c]. With FUSE
// defined, pairs of radix-2 stages run as one radix-2^2 step (4 elements per
// thread, one barrier and one threadgroup-memory round trip per pair); the
// order of operations and outputs is the same as the plain radix-2 loops.
static inline void dif_cols(threadgroup uint *s, uint L, uint ncol, device const uint *tw,
                            uint tid, uint nthr, constant Params &P) {
    uint h = L / 2;
#ifdef FUSE
    for (; h >= 2; h >>= 2) {
        uint q = h / 2;
        for (uint b = tid; b < ncol * L / 4; b += nthr) {
            uint c = b % ncol, bb = b / ncol;
            uint blk = bb / q, j = bb - blk * q;
            uint i0 = (blk * 4 * q + j) * ncol + c, i1 = i0 + q * ncol, i2 = i1 + q * ncol, i3 = i2 + q * ncol;
            uint x0 = s[i0], x1 = s[i1], x2 = s[i2], x3 = s[i3];
            uint u0 = madd(x0, x2, P.p), u1 = madd(x1, x3, P.p);
            uint u2 = mmul(msub(x0, x2, P.p), tw[j * (L / (2 * h))], P.p, P.pinv);
            uint u3 = mmul(msub(x1, x3, P.p), tw[(j + q) * (L / (2 * h))], P.p, P.pinv);
            uint w = tw[j * (L / (2 * q))];
            s[i0] = madd(u0, u1, P.p);
            s[i1] = mmul(msub(u0, u1, P.p), w, P.p, P.pinv);
            s[i2] = madd(u2, u3, P.p);
            s[i3] = mmul(msub(u2, u3, P.p), w, P.p, P.pinv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif
    for (; h >= 1; h >>= 1) {
        for (uint b = tid; b < ncol * L / 2; b += nthr) {
            uint c = b % ncol, bb = b / ncol;
            uint blk = bb / h, j = bb - blk * h;
            uint i0 = (blk * 2 * h + j) * ncol + c, i1 = i0 + h * ncol;
            uint x = s[i0], y = s[i1];
            s[i0] = madd(x, y, P.p);
            s[i1] = mmul(msub(x, y, P.p), tw[j * (L / (2 * h))], P.p, P.pinv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

static inline void dit_cols(threadgroup uint *s, uint L, uint ncol, device const uint *itw,
                            uint tid, uint nthr, constant Params &P) {
    uint h = 1;
#ifdef FUSE
    // with an odd number of stages, the single radix-2 stage goes first
    if ((31 - clz(L)) & 1) {
#endif
        for (; h <= L / 2; h <<= 1) {
#ifdef FUSE
            if (h > 1) break;
#endif
            for (uint b = tid; b < ncol * L / 2; b += nthr) {
                uint c = b % ncol, bb = b / ncol;
                uint blk = bb / h, j = bb - blk * h;
                uint i0 = (blk * 2 * h + j) * ncol + c, i1 = i0 + h * ncol;
                uint x = s[i0], y = mmul(s[i1], itw[j * (L / (2 * h))], P.p, P.pinv);
                s[i0] = madd(x, y, P.p);
                s[i1] = msub(x, y, P.p);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#ifdef FUSE
    }
    for (; h <= L / 4; h <<= 2) {
        uint q = h, hh = 2 * h;
        for (uint b = tid; b < ncol * L / 4; b += nthr) {
            uint c = b % ncol, bb = b / ncol;
            uint blk = bb / q, j = bb - blk * q;
            uint i0 = (blk * 4 * q + j) * ncol + c, i1 = i0 + q * ncol, i2 = i1 + q * ncol, i3 = i2 + q * ncol;
            uint w = itw[j * (L / (2 * q))];
            uint y1 = mmul(s[i1], w, P.p, P.pinv), y3 = mmul(s[i3], w, P.p, P.pinv);
            uint x0 = s[i0], x2 = s[i2];
            uint u0 = madd(x0, y1, P.p), u1 = msub(x0, y1, P.p);
            uint u2 = madd(x2, y3, P.p), u3 = msub(x2, y3, P.p);
            uint za = mmul(u2, itw[j * (L / (2 * hh))], P.p, P.pinv);
            uint zb = mmul(u3, itw[(j + q) * (L / (2 * hh))], P.p, P.pinv);
            s[i0] = madd(u0, za, P.p);
            s[i2] = msub(u0, za, P.p);
            s[i1] = madd(u1, zb, P.p);
            s[i3] = msub(u1, zb, P.p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif
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
    dif_cols(s, R, G, twR, tid, TPG_COL, P);
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
    dif_cols(s, C, 1, twC, tid, TPG_ROW, P);
    for (uint i = tid; i < C; i += TPG_ROW) s[i] = mmul(s[i], s[i], P.p, P.pinv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dit_cols(s, C, 1, itwC, tid, TPG_ROW, P);
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
    dit_cols(s, R, G, itwR, tid, TPG_COL, P);
    for (uint i = tid; i < R * G; i += TPG_COL) d[(i / G) * C + i % G] = s[i];
}
