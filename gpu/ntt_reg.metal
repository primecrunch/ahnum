// Prototype, register-based variant of ntt.metal: batched NTT squaring mod one
// 31-bit prime, four-step N = R x C, with each length split again as A x B:
// a thread holds B elements in registers, runs a length-B DFT there, applies
// the inner twiddle, exchanges once through threadgroup memory, and runs the
// length-A DFTs (B/A of them per thread) on the other index. Two barriers per
// row square instead of one per radix-2 stage; GPU counters showed the radix-2
// kernels stalled on threadgroup round trips (instruction utilization 21%).
//
// Index maps (length L = A * B, n = a + A * b, k = kb + B * ka):
//   step 1: thread a, DFT over b -> kb, times w_L^(a * kb)
//   step 2: per kb, DFT over a -> ka
// Host defines: R, C, G (columns per threadgroup), AR, BR (R = AR * BR),
// AC, BC (C = AC * BC), RT (rows per threadgroup in row_sq_r).
// Tables (Montgomery form, as ntt.metal): wR[i] = w_R^i (i < R), wC[i] = w_C^i
// (i < C), and their inverses; register DFTs index them with a stride.
// t4[k1 * C + n2] = w_N^(k1 * n2) (rows stored in natural frequency order);
// it4 = w_N^-(k1 * n2) * scale (scale = 2^32 / N, see ntt_proto.m).
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint p, pinv;
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

// In-register DIF of length L on x[0..L-1]: natural in, bit-reversed out.
// T is a table of w_M^i, so w_L^e = T[e * (M / L)].
template <uint L>
static inline void dif_reg(thread uint *x, constant uint *T, uint M, constant Params &P) {
#pragma clang loop unroll(full)
    for (uint h = L / 2; h >= 1; h >>= 1) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < L / 2; i++) {
            uint blk = i / h, j = i % h;
            uint i0 = blk * 2 * h + j, i1 = i0 + h;
            uint u = x[i0], v = x[i1];
            x[i0] = madd(u, v, P.p);
            x[i1] = mmul(msub(u, v, P.p), T[j * (L / (2 * h)) * (M / L)], P.p, P.pinv);
        }
    }
}

// In-register DIT of length L: bit-reversed in, natural out; T holds w_M^-i.
template <uint L>
static inline void dit_reg(thread uint *x, constant uint *T, uint M, constant Params &P) {
#pragma clang loop unroll(full)
    for (uint h = 1; h <= L / 2; h <<= 1) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < L / 2; i++) {
            uint blk = i / h, j = i % h;
            uint i0 = blk * 2 * h + j, i1 = i0 + h;
            uint u = x[i0], v = mmul(x[i1], T[j * (L / (2 * h)) * (M / L)], P.p, P.pinv);
            x[i0] = madd(u, v, P.p);
            x[i1] = msub(u, v, P.p);
        }
    }
}

static inline uint brev(uint x, uint L) { return reverse_bits(x) >> (32 - ctz(L)); }

// One threadgroup per (test, G columns); AR threads per column, thread
// (c, a) with c = tid % G. Threadgroup memory s[(kb * AR + a) * G + c].
kernel void col_fwd_r(device uint *data [[buffer(0)]], device const uint *t4 [[buffer(1)]],
                      constant Params &P [[buffer(2)]], constant uint *wR [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint s[R * G];
    uint test = tg / (C / G), cg = tg % (C / G);
    uint c = tid % G, a = tid / G, n2 = cg * G + c;
    device uint *d = data + (size_t)test * R * C + n2;
    uint x[BR];
    #pragma clang loop unroll(full)
    for (uint b = 0; b < BR; b++) x[b] = d[(a + AR * b) * C];
    dif_reg<BR>(x, wR, R, P);
    #pragma clang loop unroll(full)
    for (uint q = 0; q < BR; q++) {
        uint kb = brev(q, BR);
        s[(kb * AR + a) * G + c] = mmul(x[q], wR[a * kb], P.p, P.pinv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // step 2: this thread takes kb = a, a + AR, ... (BR / AR transforms of length AR)
    for (uint kb = a; kb < BR; kb += AR) {
        uint y[AR];
        #pragma clang loop unroll(full)
        for (uint aa = 0; aa < AR; aa++) y[aa] = s[(kb * AR + aa) * G + c];
        dif_reg<AR>(y, wR, R, P);
        #pragma clang loop unroll(full)
        for (uint q = 0; q < AR; q++) {
            uint k1 = kb + BR * brev(q, AR);
            d[k1 * C] = mmul(y[q], t4[k1 * C + n2], P.p, P.pinv);
        }
    }
}

kernel void col_inv_r(device uint *data [[buffer(0)]], device const uint *it4 [[buffer(1)]],
                      constant Params &P [[buffer(2)]], constant uint *iwR [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint s[R * G];
    uint test = tg / (C / G), cg = tg % (C / G);
    uint c = tid % G, a = tid / G, n2 = cg * G + c;
    device uint *d = data + (size_t)test * R * C + n2;
    for (uint kb = a; kb < BR; kb += AR) {
        uint y[AR];
        #pragma clang loop unroll(full)
        for (uint q = 0; q < AR; q++) {
            uint k1 = kb + BR * brev(q, AR);
            y[q] = mmul(d[k1 * C], it4[k1 * C + n2], P.p, P.pinv);
        }
        dit_reg<AR>(y, iwR, R, P);
        #pragma clang loop unroll(full)
        for (uint aa = 0; aa < AR; aa++) s[(kb * AR + aa) * G + c] = y[aa];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint x[BR];
    #pragma clang loop unroll(full)
    for (uint q = 0; q < BR; q++) {
        uint kb = brev(q, BR);
        x[q] = mmul(s[(kb * AR + a) * G + c], iwR[a * kb], P.p, P.pinv);
    }
    dit_reg<BR>(x, iwR, R, P);
    #pragma clang loop unroll(full)
    for (uint b = 0; b < BR; b++) d[(a + AR * b) * C] = x[b];
}

// RT rows per threadgroup, AC threads per row; thread (r, a) with a = tid % AC.
// Threadgroup memory s[r][kb * AC + a].
kernel void row_sq_r(device uint *data [[buffer(0)]], constant Params &P [[buffer(2)]],
                     constant uint *wC [[buffer(3)]], constant uint *iwC [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint s[RT * C];
    uint a = tid % AC, r = tid / AC;
    threadgroup uint *sr = s + r * C;
    device uint *d = data + ((size_t)tg * RT + r) * C;  // rows of all tests are contiguous
    uint x[BC];
    #pragma clang loop unroll(full)
    for (uint b = 0; b < BC; b++) x[b] = d[a + AC * b];
    dif_reg<BC>(x, wC, C, P);
    #pragma clang loop unroll(full)
    for (uint q = 0; q < BC; q++) {
        uint kb = brev(q, BC);
        sr[kb * AC + a] = mmul(x[q], wC[a * kb], P.p, P.pinv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb = a; kb < BC; kb += AC) {
        uint y[AC];
        #pragma clang loop unroll(full)
        for (uint aa = 0; aa < AC; aa++) y[aa] = sr[kb * AC + aa];
        dif_reg<AC>(y, wC, C, P);
        #pragma clang loop unroll(full)
        for (uint q = 0; q < AC; q++) y[q] = mmul(y[q], y[q], P.p, P.pinv);
        dit_reg<AC>(y, iwC, C, P);
        #pragma clang loop unroll(full)
        for (uint aa = 0; aa < AC; aa++) sr[kb * AC + aa] = y[aa];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint q = 0; q < BC; q++) {
        uint kb = brev(q, BC);
        x[q] = mmul(sr[kb * AC + a], iwC[a * kb], P.p, P.pinv);
    }
    dit_reg<BC>(x, iwC, C, P);
    #pragma clang loop unroll(full)
    for (uint b = 0; b < BC; b++) d[a + AC * b] = x[b];
}
