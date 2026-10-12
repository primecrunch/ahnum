// Prototypes for the generated assembly kernels (gen/gen_kernels.py).
#ifndef KITE_KERNELS_H
#define KITE_KERNELS_H

#include <stddef.h>

#define KITE_STAGE(dir, p)                                                                   \
    void kite_##dir##_r##p(double *base, size_t estride, size_t nlg, size_t nblocks,        \
                          size_t bstride, size_t s, const double *tw);                     \
    void kite_##dir##_r##p##_notw(double *base, size_t estride, size_t nlg, size_t nblocks, \
                                 size_t bstride, size_t s, const double *tw);              \
    void kite_##dir##_r##p##_t4(double *base, size_t estride, size_t nlg, size_t nblocks,   \
                               size_t bstride, ptrdiff_t delta);

#define KITE_STAGE_B(dir, p)                                                                    \
    void kite_##dir##_r##p##_b(double *base, size_t estride, size_t nlg, size_t nblocks,       \
                              size_t bstride, size_t s, const double *tw);                    \
    void kite_##dir##_r##p##_notw_b(double *base, size_t estride, size_t nlg, size_t nblocks,  \
                                   size_t bstride, size_t s, const double *tw);               \
    void kite_##dir##_r##p##_t4_b(double *base, size_t estride, size_t nlg, size_t nblocks,    \
                                 size_t bstride, ptrdiff_t delta);

// Tangent-form DIT stages: tw is the kite_tan_table layout (p in 2, 3, 4, 8).
#define KITE_STAGE_TAN(p)                                                                  \
    void kite_dit_r##p##_tan(double *base, size_t estride, size_t nlg, size_t nblocks,    \
                            size_t bstride, size_t s, const double *tw);                 \
    void kite_dit_r##p##_tan_b(double *base, size_t estride, size_t nlg, size_t nblocks,  \
                              size_t bstride, size_t s, const double *tw);
KITE_STAGE_TAN(2)
KITE_STAGE_TAN(3)
KITE_STAGE_TAN(4)
KITE_STAGE_TAN(8)
#undef KITE_STAGE_TAN

#define KITE_STAGES(p) KITE_STAGE(dif, p) KITE_STAGE(dit, p) KITE_STAGE_B(dif, p) KITE_STAGE_B(dit, p)
KITE_STAGES(2)
KITE_STAGES(3)
KITE_STAGES(4)
KITE_STAGES(5)
KITE_STAGES(7)
KITE_STAGES(8)
#undef KITE_STAGES
#undef KITE_STAGE
#undef KITE_STAGE_B

#include <stdint.h>
#define KITE_NORM(m)                                                                         \
    double kite_norm_##m(double *gbase, size_t rowstride, size_t npairs, double *carry,     \
                        const double *A, const double *IA, const uint64_t *mask,           \
                        const double *consts);
KITE_NORM(a)
KITE_NORM(ae)
KITE_NORM(at)
KITE_NORM(ate)
KITE_NORM(b)
KITE_NORM(be)
KITE_NORM(bt)
KITE_NORM(bte)
KITE_NORM(c)
KITE_NORM(ct)
KITE_NORM(aq)
KITE_NORM(aqe)
KITE_NORM(bq)
KITE_NORM(bqe)
KITE_NORM(cq)
KITE_NORM(ao)
KITE_NORM(aoe)
KITE_NORM(bo)
KITE_NORM(boe)
KITE_NORM(co)
#undef KITE_NORM

void kite_rows_in(const double *rp, const double *rpp, double *scr, size_t C);
void kite_rows_out(double *rp, double *rpp, const double *scr, size_t C);
// SME radix-8 stage (layout b), see gen_sme_r8. Requires FEAT_SME_F64F64.
void kite_sme_r8(double *base, size_t estride, size_t nblocks, size_t bstride, size_t s,
                const double *mats);
void kite_cmul_b(double *base, size_t estride, size_t n, ptrdiff_t delta);
void kite_cmulc_b(double *base, size_t estride, size_t n, ptrdiff_t delta);
// First/last row-FFT stage (radix 4, s = C/4) fused with the row-pair transpose.
void kite_dif_r4_rows(const double *rp, const double *rpp, double *scr, size_t C, const double *tw);
void kite_dit_r4_rows(double *rp, double *rpp, const double *scr, size_t C, const double *tw);
void kite_dit_r4_rows_tan(double *rp, double *rpp, const double *scr, size_t C, const double *tw);
void kite_split_sq(double *scr, size_t C, const double *rs, const double *cs);

#endif
