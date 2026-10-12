// ahnum: fast modular squaring modulo N = k*2^n + c for PRP testing.
//
// Numbers live in an FFT-friendly weighted representation owned by a context.
// All arithmetic is modulo N.
#ifndef AHNUM_H
#define AHNUM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AHNUM_VERSION "0.1.0"

typedef struct ahn_ctx ahn_ctx;
typedef struct ahn_num ahn_num;

typedef enum {
    AHN_BACKEND_AUTO = 0,
    AHN_BACKEND_REF = 1,   // portable scalar C (correctness oracle)
    AHN_BACKEND_NEON = 2,  // AArch64 NEON assembly
    AHN_BACKEND_SME = 3,   // M4+ SME/SME2 assembly
    AHN_BACKEND_C2P = 4,   // two-pass layout in portable C (structure reference)
} ahn_backend;

typedef struct {
    ahn_backend backend;
    uint32_t fft_len;      // 0 = choose automatically
    uint32_t threads;      // 0 or 1 = single threaded
    // FFT stage order: 0 = from the tuning file (default order if untuned),
    // else 1 + bits: bit 0 column radices ascending, bit 1 row radices ascending
    uint32_t plan_order;
} ahn_opts;

// Only c = -1 (Riesel) is supported for now.
ahn_ctx *ahn_ctx_new(uint64_t k, uint64_t n, int32_t c, const ahn_opts *o);
void ahn_ctx_free(ahn_ctx *ctx);
uint32_t ahn_fft_len(const ahn_ctx *ctx);
const char *ahn_backend_name(const ahn_ctx *ctx);

ahn_num *ahn_num_new(ahn_ctx *ctx);
void ahn_num_free(ahn_ctx *ctx, ahn_num *x);
void ahn_copy(ahn_ctx *ctx, ahn_num *dst, const ahn_num *src);
void ahn_set_u64(ahn_ctx *ctx, ahn_num *x, uint64_t v);

void ahn_square(ahn_ctx *ctx, ahn_num *x);
void ahn_square_n(ahn_ctx *ctx, ahn_num *x, uint64_t count);
void ahn_mul(ahn_ctx *ctx, ahn_num *x, const ahn_num *y);   // x = x*y
void ahn_mul_small(ahn_ctx *ctx, ahn_num *x, int64_t m);   // x = x*m, |m| < 2^20

// Largest |rounded - exact| seen since the last reset (0..0.5).
double ahn_max_roundoff(const ahn_ctx *ctx);
void ahn_reset_roundoff(ahn_ctx *ctx);
// Check round-off on every `every`-th squaring/multiplication (default 1).
void ahn_set_roundoff_check(ahn_ctx *ctx, uint32_t every);

// Fully reduce x into [0, N) as little-endian 64-bit limbs.
// `limbs` must hold ahn_limbs(ctx) words.
size_t ahn_limbs(const ahn_ctx *ctx);
void ahn_get_limbs(ahn_ctx *ctx, const ahn_num *x, uint64_t *limbs);
void ahn_set_limbs(ahn_ctx *ctx, ahn_num *x, const uint64_t *limbs);
uint64_t ahn_res64(ahn_ctx *ctx, const ahn_num *x);

// ---- Fermat PRP test ------------------------------------------------------

typedef struct {
    ahn_opts fft;              // backend / FFT length (0 = auto) / threads
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
    // test with AHN_PRP_POINT_FAILED.
    const uint32_t *points;
    uint32_t npoints;
    int (*on_point)(void *arg, uint32_t index, uint64_t iter, const uint64_t *limbs);
    void *point_arg;
    // When *abort_flag becomes non-zero the test writes a checkpoint at the last verified
    // state and returns AHN_PRP_ABORTED.
    volatile int *abort_flag;
} ahn_prp_opts;

enum { AHN_PRP_ABORTED = -5, AHN_PRP_POINT_FAILED = -6 };

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
} ahn_prp_result;

// Fermat PRP test of k*2^n-1 to the given base. Returns 0 on success (result
// in *res), negative on failure.
int ahn_prp(uint64_t k, uint64_t n, uint32_t base, const ahn_prp_opts *o, ahn_prp_result *res);

// Iteration a checkpoint written by ahn_prp for this test resumes from. 0 on success.
int ahn_prp_checkpoint_iter(const char *path, uint64_t k, uint64_t n, uint32_t base,
                            uint64_t *iter);

// Testing hook: perturb a number's internal representation.
void ahn_debug_corrupt(ahn_ctx *ctx, ahn_num *x);

#ifdef __cplusplus
}
#endif
#endif
