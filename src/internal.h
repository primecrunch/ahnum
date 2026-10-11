// Internal definitions shared by the C glue and the backends.
//
// Representation for N = k*2^n - 1 with an FFT of L real words (the k-scaled
// IBDWT; k is folded into the irrational base so the weights stay in [1, 2)):
//   T = n + log2(k),  z = 2^{T/L},  z^L = k*2^n == 1 (mod N)
//   s_j = ceil(j*T'/L) for j = 0..L, T' a rational just below T, so
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
#ifndef AHN_INTERNAL_H
#define AHN_INTERNAL_H

#include "ahnum.h"

#include <stdbool.h>
#include <stdio.h>

typedef struct ahn_backend_ops ahn_backend_ops;

struct ahn_num {
    double *d;  // backend-defined layout, ctx->num_doubles doubles
};

struct ahn_ctx {
    uint64_t k, n;
    int32_t c;
    uint32_t L;           // FFT length in real words
    uint32_t threads;
    uint8_t *bits;        // bits[j] = s_{j+1} - s_j
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
    const ahn_backend_ops *ops;
    void *bk;             // backend private state
    size_t num_doubles;   // doubles per ahn_num
    int64_t *tmpdig;      // L digits scratch
};

// Backends convert between their storage and balanced integer digits.
struct ahn_backend_ops {
    const char *name;
    int (*init)(ahn_ctx *ctx);
    void (*free)(ahn_ctx *ctx);
    void (*to_digits)(ahn_ctx *ctx, const ahn_num *x, int64_t *dig);
    void (*from_digits)(ahn_ctx *ctx, ahn_num *x, const int64_t *dig);
    void (*square)(ahn_ctx *ctx, ahn_num *x);
    void (*mul)(ahn_ctx *ctx, ahn_num *x, const ahn_num *y);
    void (*mul_small)(ahn_ctx *ctx, ahn_num *x, int64_t m);
};

extern const ahn_backend_ops ahn_ref_ops;
extern const ahn_backend_ops ahn_f2_ops;
extern const ahn_backend_ops ahn_neon_ops;

// Carry-propagate integer digits (any magnitude < 2^62) into balanced form,
// wrapping the top carry into digit 0.
void ahn_carry_digits(const ahn_ctx *ctx, int64_t *dig);

// Choose an FFT length for (k, n).
uint32_t ahn_choose_fft_len(uint64_t k, uint64_t n);
uint32_t ahn_next_fft_len(uint32_t L);
double ahn_safe_bpw(uint32_t L, uint64_t k);
bool ahn_fft_len_ok(uint32_t L);

void *ahn_alloc(size_t bytes);
void ahn_free_aligned(void *p);

#endif
