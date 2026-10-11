// Prototype: the non-NTT parts of a full GPU squaring mod N = k * 2^n - 1,
// appended to ntt_reg.metal (which supplies mmul/madd/msub, Params and the NTT
// kernels). See gpu_sq.m for the data flow. Numbers are W words of 45 bits
// (ulong), value < 2^(n + 14), in Montgomery form with R = 2^n. Since
// N = -1 mod 2^n, reduction of a product P = A + B * 2^n (A < 2^n) is
// P * 2^-n = B + k * A (mod N).
// Host defines: W (words), NP (padded length), WN0 (= n / 45), BN0 (= n % 45).

#define M45 ((1ul << 45) - 1)

struct Garner {
    uint p[4], pinv[4];
    uint c32[4];   // 2^32 mod p_i (Montgomery 1: mmul(x, c32) = x mod p, any x < 2^32)
    uint c64[4];   // 2^64 mod p_i (mmul(h, c64) = h * 2^32 mod p)
    uint inv01, inv02, inv12, inv03, inv13, inv23;  // p_i^-1 mod p_j, Montgomery form
    uint P01[2], P012[3];                           // p0 p1 and p0 p1 p2, 32-bit limbs
    uint k, pad;
};

static inline uint red32(uint x, uint p, uint pinv, uint c32) { return mmul(x, c32, p, pinv); }

// words -> residues: res[i][test * NP + j] = word j mod p_i (zero padding to NP)
kernel void to_res(device const ulong *x [[buffer(0)]], device uint *res [[buffer(1)]],
                   constant Garner &gc [[buffer(2)]], constant uint &B [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    uint test = gid / NP, j = gid % NP;
    if (test >= B) return;
    ulong v = j < W ? x[(size_t)test * W + j] : 0;
    uint lo = uint(v), hi = uint(v >> 32);
    for (uint i = 0; i < 4; i++) {
        uint r = madd(mmul(hi, gc.c64[i], gc.p[i], gc.pinv[i]), red32(lo, gc.p[i], gc.pinv[i], gc.c32[i]),
                      gc.p[i]);
        res[(size_t)i * B * NP + gid] = r;
    }
}

// 128-bit accumulate in 32-bit limbs: acc += v * c (c has n limbs)
static inline void acc_mul(thread uint *acc, uint v, constant uint *c, uint n) {
    ulong carry = 0;
    for (uint i = 0; i < 4; i++) {
        ulong t = (ulong)acc[i] + carry + (i < n ? (ulong)v * c[i] : 0);
        acc[i] = uint(t);
        carry = t >> 32;
    }
}

// Garner CRT of the 4 residues of coefficient j, split into 45-bit parts
// parts[(test * NP + j) * 3 + {0,1,2}] = bits [0,45), [45,90), [90,128).
kernel void crt_split(device const uint *res [[buffer(0)]], device ulong *parts [[buffer(1)]],
                      constant Garner &gc [[buffer(2)]], constant uint &B [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= B * NP) return;
    size_t st = (size_t)B * NP;
    uint r0 = res[gid], r1 = res[st + gid], r2 = res[2 * st + gid], r3 = res[3 * st + gid];
    // v1 = (r1 - v0) / p0 mod p1
    uint v0 = r0;
    uint t = msub(r1, red32(v0, gc.p[1], gc.pinv[1], gc.c32[1]), gc.p[1]);
    uint v1 = mmul(t, gc.inv01, gc.p[1], gc.pinv[1]);
    // v2 = ((r2 - v0) / p0 - v1) / p1 mod p2
    t = msub(r2, red32(v0, gc.p[2], gc.pinv[2], gc.c32[2]), gc.p[2]);
    t = mmul(t, gc.inv02, gc.p[2], gc.pinv[2]);
    t = msub(t, red32(v1, gc.p[2], gc.pinv[2], gc.c32[2]), gc.p[2]);
    uint v2 = mmul(t, gc.inv12, gc.p[2], gc.pinv[2]);
    // v3 = (((r3 - v0) / p0 - v1) / p1 - v2) / p2 mod p3
    t = msub(r3, red32(v0, gc.p[3], gc.pinv[3], gc.c32[3]), gc.p[3]);
    t = mmul(t, gc.inv03, gc.p[3], gc.pinv[3]);
    t = msub(t, red32(v1, gc.p[3], gc.pinv[3], gc.c32[3]), gc.p[3]);
    t = mmul(t, gc.inv13, gc.p[3], gc.pinv[3]);
    t = msub(t, red32(v2, gc.p[3], gc.pinv[3], gc.c32[3]), gc.p[3]);
    uint v3 = mmul(t, gc.inv23, gc.p[3], gc.pinv[3]);
    // c = v0 + v1 p0 + v2 p0 p1 + v3 p0 p1 p2
    uint acc[4] = {v0, 0, 0, 0};
    acc_mul(acc, v1, gc.p, 1);
    acc_mul(acc, v2, gc.P01, 2);
    acc_mul(acc, v3, gc.P012, 3);
    ulong lo = (ulong)acc[0] | ((ulong)acc[1] << 32), hi = (ulong)acc[2] | ((ulong)acc[3] << 32);
    device ulong *o = parts + (size_t)gid * 3;
    o[0] = lo & M45;
    o[1] = (lo >> 45) | ((hi & ((1ul << 26) - 1)) << 19);
    o[2] = hi >> 26;
}

// s_j = part0_j + part1_(j-1) + part2_(j-2) (< 3 * 2^45)
static inline ulong colsum(device const ulong *pt, int j) {
    ulong s = 0;
    if (j >= 0 && j < NP) s += pt[j * 3];
    if (j - 1 >= 0 && j - 1 < NP) s += pt[(j - 1) * 3 + 1];
    if (j - 2 >= 0 && j - 2 < NP) s += pt[(j - 2) * 3 + 2];
    return s;
}

// product words D_j = (s_j & M45) + (s_(j-1) >> 45), j < NP; a D_j that still
// exceeds 45 bits (a longer carry ripple) is counted in *ovf
kernel void carry(device const ulong *parts [[buffer(0)]], device ulong *D [[buffer(1)]],
                  constant uint &B [[buffer(3)]], device atomic_uint *ovf [[buffer(4)]],
                  uint gid [[thread_position_in_grid]]) {
    uint test = gid / NP;
    int j = int(gid % NP);
    if (test >= B) return;
    device const ulong *pt = parts + (size_t)test * NP * 3;
    ulong d = (colsum(pt, j) & M45) + (colsum(pt, j - 1) >> 45);
    if (d >> 45) atomic_fetch_add_explicit(ovf, 1, memory_order_relaxed);
    D[gid] = d;
}

// A_j and B_j of P = A + B 2^n, from the product words
static inline ulong wordA(device const ulong *D, int j) {
    if (j < 0) return 0;
    if (j < WN0) return D[j];
    if (j == WN0) return D[j] & ((1ul << BN0) - 1);
    return 0;
}
static inline ulong wordB(device const ulong *D, int j) {
    if (j < 0) return 0;
    int s = WN0 + j;
    if (s >= NP) return 0;
    ulong lo = D[s] >> BN0;
    ulong hi = (BN0 && s + 1 < NP) ? (D[s + 1] << (45 - BN0)) : 0;
    return (lo | hi) & M45;
}
static inline ulong wordE(device const ulong *D, int j, uint k) {
    if (j < 0) return 0;
    ulong ka = k * wordA(D, j), kp = j > 0 ? k * wordA(D, j - 1) : 0;
    return wordB(D, j) + (ka & M45) + (kp >> 45);
}

// X_j = (E_j & M45) + (E_(j-1) >> 45), E = B + k A; j < W
kernel void redc(device const ulong *D [[buffer(0)]], device ulong *x [[buffer(1)]],
                 constant Garner &gc [[buffer(2)]], constant uint &B [[buffer(3)]],
                 device atomic_uint *ovf [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint test = gid / W;
    int j = int(gid % W);
    if (test >= B) return;
    device const ulong *Dt = D + (size_t)test * NP;
    ulong v = (wordE(Dt, j, gc.k) & M45) + (wordE(Dt, j - 1, gc.k) >> 45);
    if (v >> 45) atomic_fetch_add_explicit(ovf, 1, memory_order_relaxed);
    x[gid] = v;
}

// Top fold, one thread per number: T = bits >= n; T = q k + r; X = X mod 2^n + q + r 2^n.
kernel void fold(device ulong *x [[buffer(1)]], constant Garner &gc [[buffer(2)]],
                 constant uint &B [[buffer(3)]], uint test [[thread_position_in_grid]]) {
    if (test >= B) return;
    device ulong *xt = x + (size_t)test * W;
    // carry out of the top word into T (words above W are absent: value < 2^(45 W))
    // T < 2^30 (B + k A < 2^(n + 29)), so words whose bits all lie 64 or more
    // above n are zero and are skipped
    ulong T = xt[WN0] >> BN0;
    for (int j = WN0 + 1; j < W; j++) {
        int sh = 45 * (j - WN0) - BN0;
        if (sh < 64) T |= xt[j] << sh;
    }
    ulong q = T / gc.k, r = T % gc.k;
    xt[WN0] = (xt[WN0] & ((1ul << BN0) - 1)) | (r << BN0);
    // r < 2^14 at bit BN0 < 45 can spill only into word WN0 + 1
    for (int j = WN0 + 1; j < W; j++) xt[j] = j == WN0 + 1 ? (r << BN0) >> 45 : 0;
    // r may spill past word WN0
    xt[WN0] &= M45;
    for (int j = 0; q && j < W; j++) {
        ulong s = xt[j] + q;
        xt[j] = s & M45;
        q = s >> 45;
    }
}
