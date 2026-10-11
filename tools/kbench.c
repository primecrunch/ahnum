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

int main(int argc, char **argv) {
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
