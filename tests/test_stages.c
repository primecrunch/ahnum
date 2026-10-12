// Check every generated FFT stage kernel against a scalar reference.
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

#include "internal.h"
#include "kernels.h"

typedef void (*stage_fn)(double *, size_t, size_t, size_t, size_t, size_t, const double *);
typedef void (*t4_fn)(double *, size_t, size_t, size_t, size_t, ptrdiff_t);

static double rnd(void) { return (double)rand() / RAND_MAX * 2 - 1; }

// layout a: lane-group [re l0 l1][im l0 l1] (32 bytes apart); layout b: 8-lane
// blocks, lane-group g at 16g with the imaginary parts 64 bytes on
static int g_imoff = 2, g_lgstep = 32;

static double complex get(const double *base, size_t off, int lane) {
    const double *p = (const double *)((const char *)base + off);
    return p[lane] + I * p[g_imoff + lane];
}

static void put(double *base, size_t off, int lane, double complex x) {
    double *p = (double *)((char *)base + off);
    p[lane] = creal(x);
    p[g_imoff + lane] = cimag(x);
}

static void ref_stage(double *base, int p, int dit, int mode, size_t es, size_t nlg, size_t nb,
                      size_t bs, size_t s, const double *tw, ptrdiff_t delta) {
    for (size_t b = 0; b < nb; b++)
        for (size_t j = 0; j < s; j++)
            for (size_t lg = 0; lg < nlg; lg++)
                for (int lane = 0; lane < 2; lane++) {
                    double complex x[8], y[8];
                    size_t off[8];
                    for (int t = 0; t < p; t++) {
                        off[t] = b * bs + j * es + t * s * es + lg * g_lgstep;
                        x[t] = get(base, off[t], lane);
                        if (dit && mode == 0 && t > 0) {
                            const double *w = tw + 2 * (j * (p - 1) + t - 1);
                            x[t] *= conj(w[0] + I * w[1]);
                        }
                        if (dit && mode == 2) x[t] *= conj(get(base, off[t] + delta, lane));
                    }
                    for (int u = 0; u < p; u++) {
                        double complex sacc = 0;
                        for (int t = 0; t < p; t++) {
                            double a = 2.0 * ((t * u) % p) / p;
                            sacc += x[t] * (__cospi(a) + (dit ? 1 : -1) * I * __sinpi(a));
                        }
                        y[u] = sacc;
                    }
                    for (int t = 0; t < p; t++) {
                        if (!dit && mode == 0 && t > 0) {
                            const double *w = tw + 2 * (j * (p - 1) + t - 1);
                            y[t] *= w[0] + I * w[1];
                        }
                        if (!dit && mode == 2) y[t] *= get(base, off[t] + delta, lane);
                    }
                    // write after all lanes read: lanes are independent, fine
                    for (int t = 0; t < p; t++) put(base, off[t], lane, y[t]);
                }
}

int main(void) {
    struct {
        int p;
        stage_fn f[2][2];  // [dit][notw]
        t4_fn t4[2];
    } taba[] = {
        {2, {{kite_dif_r2, kite_dif_r2_notw}, {kite_dit_r2, kite_dit_r2_notw}}, {kite_dif_r2_t4, kite_dit_r2_t4}},
        {3, {{kite_dif_r3, kite_dif_r3_notw}, {kite_dit_r3, kite_dit_r3_notw}}, {kite_dif_r3_t4, kite_dit_r3_t4}},
        {4, {{kite_dif_r4, kite_dif_r4_notw}, {kite_dit_r4, kite_dit_r4_notw}}, {kite_dif_r4_t4, kite_dit_r4_t4}},
        {5, {{kite_dif_r5, kite_dif_r5_notw}, {kite_dit_r5, kite_dit_r5_notw}}, {kite_dif_r5_t4, kite_dit_r5_t4}},
        {7, {{kite_dif_r7, kite_dif_r7_notw}, {kite_dit_r7, kite_dit_r7_notw}}, {kite_dif_r7_t4, kite_dit_r7_t4}},
        {8, {{kite_dif_r8, kite_dif_r8_notw}, {kite_dit_r8, kite_dit_r8_notw}}, {kite_dif_r8_t4, kite_dit_r8_t4}},
    }, tabb[] = {
#define ROW(P) {P, {{kite_dif_r##P##_b, kite_dif_r##P##_notw_b}, {kite_dit_r##P##_b, kite_dit_r##P##_notw_b}}, \
                {kite_dif_r##P##_t4_b, kite_dit_r##P##_t4_b}}
        ROW(2), ROW(3), ROW(4), ROW(5), ROW(7), ROW(8),
#undef ROW
    };
    int fails = 0;
    for (int layout = 0; layout < 2; layout++) {
    __typeof__(taba[0]) *tab = layout ? tabb : taba;
    g_imoff = layout ? 8 : 2;
    g_lgstep = layout ? 16 : 32;
    for (size_t i = 0; i < 6; i++) {
        int p = tab[i].p;
        for (int dit = 0; dit < 2; dit++) {
            for (int mode = 0; mode < 3; mode++) {
                size_t nlg = layout ? 4 : 3, s = mode == 0 ? 5 : 1, nb = 3;
                size_t es = layout ? 128 + 64 : 32 * nlg + 64;
                if (mode == 1) s = 4;
                size_t bs = p * s * es + 96;
                size_t bytes = nb * bs + 64;
                double *a = calloc(2 * bytes / 8 + 16, 8), *b = calloc(2 * bytes / 8 + 16, 8);
                for (size_t k = 0; k < 2 * bytes / 8; k++) a[k] = rnd();
                memcpy(b, a, 2 * bytes);
                double *tw = malloc(s * (p - 1) * 2 * sizeof(double));
                for (size_t k = 0; k < s * (p - 1); k++) {
                    double th = rnd() * 3;
                    tw[2 * k] = cos(th);
                    tw[2 * k + 1] = sin(th);
                }
                ptrdiff_t delta = (ptrdiff_t)bytes;  // t4 table lives after the data
                if (mode == 2) {
                    tab[i].t4[dit](a, es, nlg, nb, bs, delta);
                } else {
                    tab[i].f[dit][mode](a, es, nlg, nb, bs, s, tw);
                }
                ref_stage(b, p, dit, mode, es, nlg, nb, bs, s, tw, delta);
                double maxd = 0;
                for (size_t k = 0; k < bytes / 8; k++) maxd = fmax(maxd, fabs(a[k] - b[k]));
                const char *mn[] = {"tw", "notw", "t4"};
                int bad = maxd > 1e-12;
                fails += bad;
                printf("%s r%d %-4s %c %s maxdiff %.2e\n", dit ? "dit" : "dif", p, mn[mode],
                       layout ? 'b' : 'a', bad ? "FAIL" : "ok  ", maxd);
                free(a);
                free(b);
                free(tw);
            }
        }
    }
    }
    // Tangent-form DIT stages vs the reference with the matching true twiddles
    // w = e^{-2 pi i t j / (p s)}; s = 8 and 16 put cos(theta) = 0 exactly on
    // some inputs, which the table stores as c = 2^-60.
    static const struct {
        int p;
        stage_fn f[2];
    } tant[] = {{2, {kite_dit_r2_tan, kite_dit_r2_tan_b}},
                {3, {kite_dit_r3_tan, kite_dit_r3_tan_b}},
                {4, {kite_dit_r4_tan, kite_dit_r4_tan_b}},
                {8, {kite_dit_r8_tan, kite_dit_r8_tan_b}}};
    static const size_t tans[] = {5, 8, 16};
    for (int layout = 0; layout < 2; layout++) {
        g_imoff = layout ? 8 : 2;
        g_lgstep = layout ? 16 : 32;
        for (size_t i = 0; i < 4; i++) {
            for (size_t si = 0; si < 3; si++) {
                int p = tant[i].p;
                size_t s = tans[si], nlg = layout ? 4 : 3, nb = 3;
                size_t es = layout ? 128 + 64 : 32 * nlg + 64;
                size_t bs = p * s * es + 96, bytes = nb * bs + 64;
                double *a = calloc(bytes / 8 + 16, 8), *b = calloc(bytes / 8 + 16, 8);
                for (size_t k = 0; k < bytes / 8; k++) a[k] = rnd();
                memcpy(b, a, bytes);
                double *tw = malloc(s * (p - 1) * 2 * sizeof(double));
                for (size_t j = 0; j < s; j++) {
                    for (int t = 1; t < p; t++) {
                        double ang = 2.0 * (double)((t * j) % (p * s)) / (double)(p * s);
                        tw[2 * (j * (p - 1) + t - 1)] = __cospi(ang);
                        tw[2 * (j * (p - 1) + t - 1) + 1] = -__sinpi(ang);
                    }
                }
                double *tt = malloc(s * kite_tan_nscal(p) * sizeof(double));
                kite_tan_table(p, s, tt);
                tant[i].f[layout](a, es, nlg, nb, bs, s, tt);
                ref_stage(b, p, 1, 0, es, nlg, nb, bs, s, tw, 0);
                double maxd = 0;
                for (size_t k = 0; k < bytes / 8; k++) maxd = fmax(maxd, fabs(a[k] - b[k]));
                int bad = maxd > 1e-12;
                fails += bad;
                printf("dit r%d tan  %c s=%-2zu %s maxdiff %.2e\n", p, layout ? 'b' : 'a', s,
                       bad ? "FAIL" : "ok  ", maxd);
                free(a);
                free(b);
                free(tw);
                free(tt);
            }
        }
    }
    // SME radix-8 stage vs the reference, layout b, both directions (M4 and later; the
    // kernel is an illegal instruction elsewhere, so check as the SME backend does)
    int sme = 0;
    size_t sme_len = sizeof sme;
    if (sysctlbyname("hw.optional.arm.FEAT_SME_F64F64", &sme, &sme_len, NULL, 0)) sme = 0;
    if (!sme) printf("sme r8: skipped, no SME on this CPU\n");
    g_imoff = 8;
    g_lgstep = 16;
    for (int dit = 0; dit < 2 && sme; dit++) {
        for (int sv = 0; sv < 2; sv++) {
            size_t s = sv ? 5 : 2, nb = sv ? 2 : 3, es = 128 + 64, nlg = 4;
            size_t bs = 8 * s * es + 128, bytes = nb * bs + 128;
            double *a = calloc(bytes / 8 + 16, 8), *b = calloc(bytes / 8 + 16, 8);
            for (size_t k = 0; k < bytes / 8; k++) a[k] = rnd();
            memcpy(b, a, bytes);
            double *tw = malloc(s * 7 * 2 * sizeof(double));
            for (size_t k = 0; k < s * 7; k++) {
                double th = rnd() * 3;
                tw[2 * k] = cos(th);
                tw[2 * k + 1] = sin(th);
            }
            double *mats = malloc(s * 128 * sizeof(double));
            for (size_t j = 0; j < s; j++) {
                for (int u = 0; u < 8; u++) {
                    for (int t = 0; t < 8; t++) {
                        double ang = 2.0 * ((t * u) % 8) / 8;
                        double complex f = __cospi(ang) - I * __sinpi(ang);
                        double complex w;
                        if (!dit) {
                            w = t ? tw[2 * (j * 7 + t - 1)] + I * tw[2 * (j * 7 + t - 1) + 1] : 1;
                            f *= w;
                        } else {
                            w = u ? tw[2 * (j * 7 + u - 1)] + I * tw[2 * (j * 7 + u - 1) + 1] : 1;
                            f = conj(f) * conj(w);
                        }
                        mats[j * 128 + u * 16 + t] = creal(f);
                        mats[j * 128 + u * 16 + 8 + t] = cimag(f);
                    }
                }
            }
            kite_sme_r8(a, es, nb, bs, s, mats);
            ref_stage(b, 8, dit, 0, es, nlg, nb, bs, s, tw, 0);
            double maxd = 0;
            for (size_t k = 0; k < bytes / 8; k++) maxd = fmax(maxd, fabs(a[k] - b[k]));
            int bad = maxd > 1e-12;
            fails += bad;
            printf("sme r8 %s s=%zu nb=%zu %s maxdiff %.2e\n", dit ? "dit" : "dif", s, nb,
                   bad ? "FAIL" : "ok  ", maxd);
            free(a);
            free(b);
            free(tw);
            free(mats);
        }
    }
    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails != 0;
}
