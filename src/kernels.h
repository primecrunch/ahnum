// Prototypes for the generated assembly kernels (gen/gen_kernels.py).
#ifndef AHN_KERNELS_H
#define AHN_KERNELS_H

#include <stddef.h>

#define AHN_STAGE(dir, p)                                                                   \
    void ahn_##dir##_r##p(double *base, size_t estride, size_t nlg, size_t nblocks,        \
                          size_t bstride, size_t s, const double *tw);                     \
    void ahn_##dir##_r##p##_notw(double *base, size_t estride, size_t nlg, size_t nblocks, \
                                 size_t bstride, size_t s, const double *tw);              \
    void ahn_##dir##_r##p##_t4(double *base, size_t estride, size_t nlg, size_t nblocks,   \
                               size_t bstride, ptrdiff_t delta);

#define AHN_STAGE_B(dir, p)                                                                    \
    void ahn_##dir##_r##p##_b(double *base, size_t estride, size_t nlg, size_t nblocks,       \
                              size_t bstride, size_t s, const double *tw);                    \
    void ahn_##dir##_r##p##_notw_b(double *base, size_t estride, size_t nlg, size_t nblocks,  \
                                   size_t bstride, size_t s, const double *tw);               \
    void ahn_##dir##_r##p##_t4_b(double *base, size_t estride, size_t nlg, size_t nblocks,    \
                                 size_t bstride, ptrdiff_t delta);

#define AHN_STAGES(p) AHN_STAGE(dif, p) AHN_STAGE(dit, p) AHN_STAGE_B(dif, p) AHN_STAGE_B(dit, p)
AHN_STAGES(2)
AHN_STAGES(3)
AHN_STAGES(4)
AHN_STAGES(5)
AHN_STAGES(7)
AHN_STAGES(8)
#undef AHN_STAGES
#undef AHN_STAGE
#undef AHN_STAGE_B

#include <stdint.h>
#define AHN_NORM(m)                                                                         \
    double ahn_norm_##m(double *gbase, size_t rowstride, size_t npairs, double *carry,     \
                        const double *A, const double *IA, const uint64_t *mask,           \
                        const double *consts);
AHN_NORM(a)
AHN_NORM(ae)
AHN_NORM(at)
AHN_NORM(ate)
AHN_NORM(b)
AHN_NORM(be)
AHN_NORM(bt)
AHN_NORM(bte)
AHN_NORM(c)
AHN_NORM(ct)
AHN_NORM(aq)
AHN_NORM(aqe)
AHN_NORM(bq)
AHN_NORM(bqe)
AHN_NORM(cq)
#undef AHN_NORM

void ahn_rows_in(const double *rp, const double *rpp, double *scr, size_t C);
void ahn_rows_out(double *rp, double *rpp, const double *scr, size_t C);
// SME radix-8 stage (layout b), see gen_sme_r8. Requires FEAT_SME_F64F64.
void ahn_sme_r8(double *base, size_t estride, size_t nblocks, size_t bstride, size_t s,
                const double *mats);
void ahn_cmul_b(double *base, size_t estride, size_t n, ptrdiff_t delta);
void ahn_cmulc_b(double *base, size_t estride, size_t n, ptrdiff_t delta);
// First/last row-FFT stage (radix 4, s = C/4) fused with the row-pair transpose.
void ahn_dif_r4_rows(const double *rp, const double *rpp, double *scr, size_t C, const double *tw);
void ahn_dit_r4_rows(double *rp, double *rpp, const double *scr, size_t C, const double *tw);
void ahn_split_sq(double *scr, size_t C, const double *rs, const double *cs);

#endif
