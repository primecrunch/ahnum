// Measure FP64 FMA throughput of NEON, streaming SVE and SME outer products,
// single-threaded and with several threads (the SME unit is shared per cluster).
#include <pthread.h>
#include <pthread/qos.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void probe_neon_fmla(uint64_t iters);  // 16 x 2-lane FMA per iter
void probe_ssve_fmla(uint64_t iters);  // 8 x SVL/8-lane FMA per iter
void probe_sme_fmopa(uint64_t iters);  // 8 x (SVL/8)^2 FMA per iter
uint64_t probe_svl_bytes(void);

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    int which;
    uint64_t iters;
    double secs;
} arg_t;

static void *run(void *p) {
    arg_t *a = p;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    double t0 = now();
    if (a->which == 0) probe_neon_fmla(a->iters);
    else if (a->which == 1) probe_ssve_fmla(a->iters);
    else probe_sme_fmopa(a->iters);
    a->secs = now() - t0;
    return NULL;
}

int main(void) {
    uint64_t svl = probe_svl_bytes(), lanes = svl / 8;
    printf("streaming vector length: %llu bits (%llu doubles)\n", (unsigned long long)svl * 8,
           (unsigned long long)lanes);
    const char *names[] = {"NEON fmla.2d", "SSVE fmla z.d", "SME fmopa za.d"};
    double fma_per_iter[] = {32, 8.0 * lanes, 8.0 * lanes * lanes};
    uint64_t iters[] = {200000000, 20000000, 2000000};
    int threads[] = {1, 2, 5, 10};
    for (int w = 0; w < 3; w++) {
        for (int ti = 0; ti < 4; ti++) {
            int T = threads[ti];
            pthread_t th[16];
            arg_t a[16];
            double t0 = now();
            for (int i = 0; i < T; i++) {
                a[i] = (arg_t){w, iters[w], 0};
                pthread_create(&th[i], NULL, run, &a[i]);
            }
            for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
            double wall = now() - t0;
            double gflops = 2.0 * fma_per_iter[w] * iters[w] * T / wall / 1e9;
            printf("%-15s threads=%-2d %8.1f GFLOPS FP64  (%.1f per thread)\n", names[w], T,
                   gflops, gflops / T);
        }
    }
    return 0;
}
