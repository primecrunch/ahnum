// Check every generated FFT stage kernel against a scalar reference.
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"

typedef void (*stage_fn)(double *, size_t, size_t, size_t, size_t, size_t, const double *);
typedef void (*t4_fn)(double *, size_t, size_t, size_t, size_t, ptrdiff_t);

static double rnd(void) { return (double)rand() / RAND_MAX * 2 - 1; }

static double complex get(const double *base, size_t off, int lane) {
    const double *p = (const double *)((const char *)base + off);
    return p[lane] + I * p[2 + lane];
}

static void put(double *base, size_t off, int lane, double complex x) {
    double *p = (double *)((char *)base + off);
    p[lane] = creal(x);
    p[2 + lane] = cimag(x);
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
                        off[t] = b * bs + j * es + t * s * es + lg * 32;
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
    } tab[] = {
        {2, {{ahn_dif_r2, ahn_dif_r2_notw}, {ahn_dit_r2, ahn_dit_r2_notw}}, {ahn_dif_r2_t4, ahn_dit_r2_t4}},
        {3, {{ahn_dif_r3, ahn_dif_r3_notw}, {ahn_dit_r3, ahn_dit_r3_notw}}, {ahn_dif_r3_t4, ahn_dit_r3_t4}},
        {4, {{ahn_dif_r4, ahn_dif_r4_notw}, {ahn_dit_r4, ahn_dit_r4_notw}}, {ahn_dif_r4_t4, ahn_dit_r4_t4}},
        {5, {{ahn_dif_r5, ahn_dif_r5_notw}, {ahn_dit_r5, ahn_dit_r5_notw}}, {ahn_dif_r5_t4, ahn_dit_r5_t4}},
        {7, {{ahn_dif_r7, ahn_dif_r7_notw}, {ahn_dit_r7, ahn_dit_r7_notw}}, {ahn_dif_r7_t4, ahn_dit_r7_t4}},
        {8, {{ahn_dif_r8, ahn_dif_r8_notw}, {ahn_dit_r8, ahn_dit_r8_notw}}, {ahn_dif_r8_t4, ahn_dit_r8_t4}},
    };
    int fails = 0;
    for (size_t i = 0; i < sizeof tab / sizeof tab[0]; i++) {
        int p = tab[i].p;
        for (int dit = 0; dit < 2; dit++) {
            for (int mode = 0; mode < 3; mode++) {
                size_t nlg = 3, s = mode == 0 ? 5 : 1, es = 32 * nlg + 64, nb = 3;
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
                printf("%s r%d %-4s %s maxdiff %.2e\n", dit ? "dit" : "dif", p, mn[mode],
                       bad ? "FAIL" : "ok  ", maxd);
                free(a);
                free(b);
                free(tw);
            }
        }
    }
    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails != 0;
}
