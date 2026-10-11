// Micro-benchmark individual kernels on L1-resident data: ns per butterfly
// (or per element) and implied instructions/cycle given a static count.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernels.h"

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef void (*stage_fn)(double *, size_t, size_t, size_t, size_t, size_t, const double *);
typedef void (*t4_fn)(double *, size_t, size_t, size_t, size_t, ptrdiff_t);

static double ghz = 4.4;

static void *amalloc(size_t n) {
    void *p = NULL;
    posix_memalign(&p, 128, n);
    return p;
}

static void bench_stage(const char *name, stage_fn f, t4_fn f4, int p, size_t nlg, size_t s,
                        int fp_instr) {
    // n elements of nlg lane-groups: total bytes ~ 32KB
    size_t es = 32 * nlg, n = 32768 / es;
    n -= n % (p * s);
    size_t nb = n / (p * s), bs = p * s * es;
    double *d = amalloc(2 * n * es + 4096);
    for (size_t i = 0; i < 2 * n * es / 8; i++) d[i] = 1e-3 * (double)(i % 97);
    double *tw = amalloc(s * p * 16 + 64);
    for (size_t i = 0; i < s * p * 2; i++) tw[i] = 0.7;
    size_t bflies = nb * s * nlg;
    int reps = 20000;
    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        for (int i = 0; i < reps / 5; i++) {
            if (f) f(d, es, nlg, nb, bs, s, tw);
            else f4(d, es, nlg, nb, bs, (ptrdiff_t)(n * es));
        }
        double dt = (now() - t0) / (reps / 5);
        if (dt < best) best = dt;
    }
    double ns = best * 1e9 / bflies;
    double cyc = ns * ghz;
    printf("%-14s nlg=%zu s=%-4zu %6.2f ns/bfly  %6.1f cyc/bfly  IPC(fp) %.2f\n", name, nlg, s, ns,
           cyc, fp_instr / cyc);
    free(d);
    free(tw);
}

// SME radix-8 vs NEON radix-8 on layout b: R elements x 8 lanes, all stages of
// size R handled as one stage with s = R/8 (worst case: one item per j).
static void bench_sme(size_t R, size_t s) {
    size_t es = 128, nb = R / (8 * s), bs = 8 * s * es;
    double *d = amalloc(R * es + 4096), *m = amalloc(s * 1024 + 128), *tw = amalloc(s * 7 * 16 + 64);
    for (size_t i = 0; i < R * es / 8; i++) d[i] = 1e-3 * (double)(i % 97);
    for (size_t i = 0; i < s * 128; i++) m[i] = 0.1;
    for (size_t i = 0; i < s * 14; i++) tw[i] = 0.7;
    size_t pts = R * 8;
    int reps = 2000;
    double bs_sme = 1e30, bs_neon = 1e30;
    for (int r = 0; r < 5; r++) {
        double t0 = now();
        for (int i = 0; i < reps; i++) ahn_sme_r8(d, es, nb, bs, s, m);
        double t = (now() - t0) / reps;
        if (t < bs_sme) bs_sme = t;
        t0 = now();
        for (int i = 0; i < reps; i++) ahn_dif_r8_b(d, es, 4, nb, bs, s, tw);
        t = (now() - t0) / reps;
        if (t < bs_neon) bs_neon = t;
    }
    printf("R=%-5zu s=%-4zu (%4zu KB)  SME %.3f cyc/pt  NEON %.3f cyc/pt  ratio %.2fx\n", R, s,
           R * es / 1024, bs_sme * ghz * 1e9 / pts, bs_neon * ghz * 1e9 / pts, bs_neon / bs_sme);
    free(d);
    free(m);
    free(tw);
}

// SME stage alternating with a NEON pass over the same data (the real pass-1
// pattern) vs NEON only.
static void bench_handoff(size_t R) {
    size_t es = 128, s = 1, nb = R / 8, bs = 8 * es;
    double *d = amalloc(2 * R * es + 4096), *m = amalloc(1024 + 128), *tw = amalloc(1024);
    for (size_t i = 0; i < 2 * R * es / 8; i++) d[i] = 1e-3 * (double)(i % 97);
    for (size_t i = 0; i < 128; i++) m[i] = 0.1;
    int reps = 4000;
    double t0 = now();
    for (int i = 0; i < reps; i++) ahn_sme_r8(d, es, nb, bs, s, m);
    double t_sme = (now() - t0) / reps;
    t0 = now();
    for (int i = 0; i < reps; i++) ahn_cmul_b(d, es, R, (ptrdiff_t)(R * es));
    double t_neon = (now() - t0) / reps;
    t0 = now();
    for (int i = 0; i < reps; i++) {
        ahn_sme_r8(d, es, nb, bs, s, m);
        ahn_cmul_b(d, es, R, (ptrdiff_t)(R * es));
    }
    double t_both = (now() - t0) / reps;
    printf("R=%-5zu (%4zu KB) SME %.0f ns  NEON pass %.0f ns  alternating %.0f ns (sum %.0f)\n", R,
           R * es / 1024, t_sme * 1e9, t_neon * 1e9, t_both * 1e9, (t_sme + t_neon) * 1e9);
    free(d);
    free(m);
    free(tw);
}

int main(int argc, char **argv) {
    if (argc > 2 && !strcmp(argv[2], "handoff")) {
        for (size_t R = 64; R <= 4096; R *= 4) bench_handoff(R);
        return 0;
    }
    if (argc > 2) {
        for (size_t R = 64; R <= 8192; R *= 2) {
            bench_sme(R, R / 8);
            bench_sme(R, 1);
        }
        return 0;
    }
    if (argc > 1) ghz = atof(argv[1]);
    // FP instruction counts per butterfly (2 lanes): DFT + twiddles
    bench_stage("dif_r8", ahn_dif_r8, NULL, 8, 1, 8, 56 + 28);
    bench_stage("dif_r8", ahn_dif_r8, NULL, 8, 4, 8, 56 + 28);
    bench_stage("dif_r8_notw", ahn_dif_r8_notw, NULL, 8, 1, 1, 56);
    bench_stage("dif_r8_t4", NULL, ahn_dif_r8_t4, 8, 4, 1, 56 + 32);
    bench_stage("dit_r8", ahn_dit_r8, NULL, 8, 1, 8, 56 + 28);
    bench_stage("dif_r4", ahn_dif_r4, NULL, 4, 1, 8, 16 + 12);
    bench_stage("dif_r4_notw", ahn_dif_r4_notw, NULL, 4, 1, 1, 16);
    bench_stage("dif_r7_notw", ahn_dif_r7_notw, NULL, 7, 1, 1, 72);
    bench_stage("dif_r7", ahn_dif_r7, NULL, 7, 1, 8, 72 + 24);
    bench_stage("dif_r5_notw", ahn_dif_r5_notw, NULL, 5, 1, 1, 40);
    bench_stage("dif_r3_notw", ahn_dif_r3_notw, NULL, 3, 1, 1, 20);
    bench_stage("dif_r2", ahn_dif_r2, NULL, 2, 1, 8, 4 + 4);
    return 0;
}
