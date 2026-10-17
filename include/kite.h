// kite: fast modular squaring modulo N = k*2^n + c for PRP testing.
//
// Numbers live in an FFT-friendly weighted representation owned by a context.
// All arithmetic is modulo N.
#ifndef KITE_H
#define KITE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KITE_VERSION "0.1.5"

typedef struct kite_ctx kite_ctx;
typedef struct kite_num kite_num;

typedef enum {
    KITE_BACKEND_AUTO = 0,
    KITE_BACKEND_REF = 1,   // portable scalar C (correctness oracle)
    KITE_BACKEND_NEON = 2,  // AArch64 NEON assembly
    KITE_BACKEND_SME = 3,   // M4+ SME/SME2 assembly
    KITE_BACKEND_C2P = 4,   // two-pass layout in portable C (structure reference)
} kite_backend;

typedef struct {
    kite_backend backend;
    uint32_t fft_len;      // 0 = choose automatically
    uint32_t threads;      // 0 or 1 = single threaded
    // FFT stage order: 0 = from the tuning file (default order if untuned),
    // else 1 + bits: bit 0 column radices ascending, bit 1 row radices ascending
    uint32_t plan_order;
} kite_opts;

// Only c = -1 (Riesel) is supported for now.
kite_ctx *kite_ctx_new(uint64_t k, uint64_t n, int32_t c, const kite_opts *o);
void kite_ctx_free(kite_ctx *ctx);
uint32_t kite_fft_len(const kite_ctx *ctx);
const char *kite_backend_name(const kite_ctx *ctx);

kite_num *kite_num_new(kite_ctx *ctx);
void kite_num_free(kite_ctx *ctx, kite_num *x);
void kite_copy(kite_ctx *ctx, kite_num *dst, const kite_num *src);
void kite_set_u64(kite_ctx *ctx, kite_num *x, uint64_t v);

void kite_square(kite_ctx *ctx, kite_num *x);
void kite_square_n(kite_ctx *ctx, kite_num *x, uint64_t count);
void kite_mul(kite_ctx *ctx, kite_num *x, const kite_num *y);   // x = x*y
void kite_mul_small(kite_ctx *ctx, kite_num *x, int64_t m);   // x = x*m, |m| < 2^20
// x = x + y and x = x - y. The stored form is linear in the value, so these are element-wise;
// the result's digits are left unbalanced until the next square or mul renormalises them,
// so do not chain more than one add or sub between multiplications.
void kite_add(kite_ctx *ctx, kite_num *x, const kite_num *y);
void kite_sub(kite_ctx *ctx, kite_num *x, const kite_num *y);

// Largest |rounded - exact| seen since the last reset (0..0.5).
double kite_max_roundoff(const kite_ctx *ctx);
void kite_reset_roundoff(kite_ctx *ctx);
// Check round-off on every `every`-th squaring/multiplication (default 1).
void kite_set_roundoff_check(kite_ctx *ctx, uint32_t every);

// Fully reduce x into [0, N) as little-endian 64-bit limbs.
// `limbs` must hold kite_limbs(ctx) words.
size_t kite_limbs(const kite_ctx *ctx);
void kite_get_limbs(kite_ctx *ctx, const kite_num *x, uint64_t *limbs);
void kite_set_limbs(kite_ctx *ctx, kite_num *x, const uint64_t *limbs);
uint64_t kite_res64(kite_ctx *ctx, const kite_num *x);

// ---- Fermat PRP test ------------------------------------------------------

typedef struct {
    kite_opts fft;              // backend / FFT length (0 = auto) / threads
    uint32_t gerbicz_block;    // 0 = auto (~sqrt(n), at least 32)
    uint32_t gerbicz_blocks;   // blocks between checks; 0 = auto (block/4, at least 8)
    double max_roundoff;       // 0 = 0.40; above this the step is redone with a larger FFT
    const char *checkpoint;    // save file path (NULL = none)
    double checkpoint_secs;    // 0 = 600
    void (*progress)(void *arg, uint64_t iter, uint64_t n, double maxerr);
    void *progress_arg;
    uint64_t inject_fault_at;  // testing: corrupt the state at this iteration (0 = never)
    // Proof points: ascending iteration positions (0..n) at which x_i is reported through
    // on_point, only once a Gerbicz check (or the doubled tail) has verified it. Points at
    // or before a resumed checkpoint are not reported again. A non-zero return stops the
    // test with KITE_PRP_POINT_FAILED.
    const uint32_t *points;
    uint32_t npoints;
    int (*on_point)(void *arg, uint32_t index, uint64_t iter, const uint64_t *limbs);
    void *point_arg;
    // When *abort_flag becomes non-zero the test writes a checkpoint at the last verified
    // state and returns KITE_PRP_ABORTED.
    volatile int *abort_flag;
} kite_prp_opts;

enum { KITE_PRP_ABORTED = -5, KITE_PRP_POINT_FAILED = -6 };

typedef struct {
    uint64_t res64;            // low 64 bits of base^(N-1) mod N
    int is_prp;
    uint32_t fft_len;          // final FFT length
    double maxerr;             // max round-off seen
    uint32_t gerbicz_errors;   // detected and recovered
    uint32_t roundoff_retries; // FFT length increases
    int resumed;               // started from a checkpoint
    uint64_t resumed_at;       // iteration it resumed from
    double seconds;
} kite_prp_result;

// Fermat PRP test of k*2^n-1 to the given base. Returns 0 on success (result
// in *res), negative on failure.
int kite_prp(uint64_t k, uint64_t n, uint32_t base, const kite_prp_opts *o, kite_prp_result *res);

// ---- LLR (Morrison) test, as PRST computes it ------------------------------------

typedef struct {
    kite_opts fft;             // backend / FFT length (0 = auto) / threads
    double max_roundoff;       // 0 = 0.40
    const char *checkpoint;    // save file path (NULL = none)
    double checkpoint_secs;    // 0 = 600
    void (*progress)(void *arg, uint64_t iter, uint64_t total, double maxerr);
    void (*on_confirm)(void *arg, int run);  // a composite result is being confirmed
    void *progress_arg;
    volatile int *abort_flag;  // as for kite_prp; returns KITE_LLR_ABORTED
    // Testing hooks. A fault corrupts the value after doubling inject_fault_at (1-based, 0 =
    // never) in each run whose bit is set in inject_fault_runs (0 = run 0 only). A round-off
    // fault makes run 0's first check at or after doubling inject_roundoff_at (or the check
    // after the Lucas ladder, with inject_roundoff_ladder) fail once.
    uint64_t inject_fault_at;
    uint32_t inject_fault_runs;
    uint64_t inject_roundoff_at;
    int inject_roundoff_ladder;
} kite_llr_opts;

typedef struct {
    int is_prime;              // N is prime
    uint64_t factor;           // non-zero: a small factor found while choosing P
    uint64_t res64;            // low 64 bits of V_{k*2^(n-1)}(P, -1) when composite
    uint32_t P;
    uint32_t fft_len;
    double maxerr;
    uint32_t roundoff_retries;
    uint32_t confirm_runs;     // extra complete runs made to confirm a composite result
    int resumed;
    uint64_t resumed_at;
    double seconds;
} kite_llr_result;

enum { KITE_LLR_ABORTED = -5 };

// LLR test of k*2^n-1 (k odd, n >= 3) with PRST's choice of P and Q = -1. 0 on success.
int kite_llr(uint64_t k, uint64_t n, const kite_llr_opts *o, kite_llr_result *res);
// P as PRST chooses it, and a factor of N found from it (0 if none).
int kite_llr_choose_P(uint64_t k, uint64_t n, uint32_t *P, uint64_t *factor);
// Iteration and run (0 = first, 1-2 = confirmation) a checkpoint resumes from; run may be NULL.
int kite_llr_checkpoint_iter(const char *path, uint64_t k, uint64_t n, uint64_t *iter, int *run);

// Iteration a checkpoint written by kite_prp for this test resumes from. 0 on success.
int kite_prp_checkpoint_iter(const char *path, uint64_t k, uint64_t n, uint32_t base,
                            uint64_t *iter);

// Testing hook: perturb a number's internal representation.
void kite_debug_corrupt(kite_ctx *ctx, kite_num *x);

#ifdef __cplusplus
}
#endif
#endif
