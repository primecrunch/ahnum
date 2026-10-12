// Internal definitions shared by the C glue and the backends.
//
// Representation for N = k*2^n - 1 with an FFT of L real words (the k-scaled
// IBDWT; k is folded into the irrational base so the weights stay in [1, 2)):
//   T = n + log2(k),  z = 2^{T/L},  z^L = k*2^n == 1 (mod N)
//   s_j = ceil(j*T'/L) for j = 0..L, T' = n + log2(k) to 64 fractional bits, so
//   s_L = n + ceil(log2 k) (s_L = n when k = 1)
//   value V = sum_j x_j 2^{s_j}; digit j has base 2^{s_{j+1}-s_j}.
//   After carrying, the part of V at or above 2^n (held in the top digits
//   t..L-1 plus the final carry) is "folded": H*2^n with H = q*k + r becomes
//   q + r*2^n since k*2^n == 1 (mod N).
//   weights w_j = 2^{s_j - j*T/L} in [1, 2)
// The weighted cyclic convolution yields, per output word, an integer plus
// (wrapped part)/k, so outputs are multiplied by k before rounding. Squaring V
// therefore produces k*V^2. To make that consistent the stored value is
// V = x/k (mod N), i.e. x*2^n: S(x)^2 * k = S(x^2).
// Multiplication by a small constant is done on the digits without the k.
#ifndef KITE_INTERNAL_H
#define KITE_INTERNAL_H

#include "kite.h"

#include <stdbool.h>
#include <stdio.h>

typedef struct kite_backend_ops kite_backend_ops;

struct kite_num {
    double *d;  // backend-defined layout, ctx->num_doubles doubles
};

struct kite_ctx {
    uint64_t k, n;
    int32_t c;
    uint32_t L;           // FFT length in real words
    uint32_t threads;
    uint32_t plan_order;  // bit 0: column radices ascending, bit 1: row radices ascending
    uint8_t *bits;        // bits[j] = s_{j+1} - s_j
    uint64_t lam;         // log2(k) as a 64-bit binary fraction (T' = n + lam/2^64)
    uint64_t sL;          // s_L
    uint32_t top_t;       // first digit with s_{t+1} > n (L if none)
    uint64_t top_st;      // s_t
    uint32_t top_m;       // n - s_t
    double *w;            // w[j]
    double *oscale;       // k / w[j]: output unweight-and-scale factor
    size_t nlimbs;        // limbs for values in [0, N)
    uint64_t *Nlimbs;     // N itself
    double maxerr;
    uint32_t err_every;   // check round-off every N-th operation (1 = always)
    uint64_t op_count;
    const kite_backend_ops *ops;
    void *bk;             // backend private state
    size_t num_doubles;   // doubles per kite_num
    int64_t *tmpdig;      // L digits scratch
};

// Backends convert between their storage and balanced integer digits.
struct kite_backend_ops {
    const char *name;
    int (*init)(kite_ctx *ctx);
    void (*free)(kite_ctx *ctx);
    void (*to_digits)(kite_ctx *ctx, const kite_num *x, int64_t *dig);
    void (*from_digits)(kite_ctx *ctx, kite_num *x, const int64_t *dig);
    void (*square)(kite_ctx *ctx, kite_num *x);
    void (*mul)(kite_ctx *ctx, kite_num *x, const kite_num *y);
    void (*mul_small)(kite_ctx *ctx, kite_num *x, int64_t m);
};

extern const kite_backend_ops kite_ref_ops;
extern const kite_backend_ops kite_f2_ops;
extern const kite_backend_ops kite_neon_ops;
extern const kite_backend_ops kite_sme_ops;

// Carry-propagate integer digits (any magnitude < 2^62) into balanced form,
// wrapping the top carry into digit 0.
void kite_carry_digits(const kite_ctx *ctx, int64_t *dig);

// Choose an FFT length for (k, n).
uint32_t kite_choose_fft_len(uint64_t k, uint64_t n);
uint32_t kite_next_fft_len(uint32_t L);
uint32_t kite_tuned_plan_order(uint32_t L);
// y = base^(N+1) mod N (n limbs) -> base^(N-1) mod N, also when gcd(base, N) > 1
// (src/prp.c); returns -1 if the base-prime part of N does not fit a word.
int kite_residue_nm1(uint64_t *y, const uint64_t *N, size_t n, uint64_t base);
double kite_safe_bpw(uint32_t L, uint64_t k);
// Layout helpers shared by the backends (exact rational arithmetic in T').
uint64_t kite_s_at(const kite_ctx *ctx, uint64_t j);
double kite_weight(const kite_ctx *ctx, uint64_t j, uint64_t sj);
bool kite_fft_len_ok(uint32_t L);
// Tangent-form DIT twiddles for a radix-p stage of block size p*s (p in 2, 3, 4,
// 8): writes s records of kite_tan_nscal(p) doubles; returns 0, or -1 for other p.
int kite_tan_table(uint32_t p, uint32_t s, double *out);
uint32_t kite_tan_nscal(uint32_t p);

void *kite_alloc(size_t bytes);
void kite_free_aligned(void *p);

#endif
