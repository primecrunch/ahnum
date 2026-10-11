// ahbench: Fermat PRP tests, timing and tuning for Riesel numbers k*2^n-1.
//
//   ahbench prp  K N | "K*2^N-1"  [--expect HEX] [--base B] [--ckpt FILE]
//                                 [--ckpt-secs S] [--block B] [--fault ITER]
//   ahbench time K N [--iters N]
//   ahbench throughput K N [--configs 10x1,5x2,...] [--secs S]
//   ahbench tune LMIN LMAX [--out FILE]
// Common options: --backend ref|c2p|neon  --fft L  --threads T
//
// `prp` runs the library driver (Gerbicz-checked) and prints RES64, the low 64
// bits of base^(N-1) mod N in hex; with --expect it exits non-zero on mismatch.
#include <inttypes.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ahnum.h"

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static ahn_backend parse_backend(const char *s) {
    if (!strcmp(s, "ref")) return AHN_BACKEND_REF;
    if (!strcmp(s, "c2p")) return AHN_BACKEND_C2P;
    if (!strcmp(s, "neon")) return AHN_BACKEND_NEON;
    if (!strcmp(s, "sme")) return AHN_BACKEND_SME;
    return AHN_BACKEND_AUTO;
}

// Accept "K N" or "K*2^N-1" (the latter as one argument). Returns the number
// of argv entries consumed, 0 on error.
static int parse_number(int argc, char **argv, int i, uint64_t *k, uint64_t *n) {
    if (i < argc && strchr(argv[i], '*')) {
        unsigned long long kk, nn;
        if (sscanf(argv[i], "%llu*2^%llu-1", &kk, &nn) != 2) return 0;
        *k = kk;
        *n = nn;
        return 1;
    }
    if (i + 1 >= argc) return 0;
    *k = strtoull(argv[i], 0, 10);
    *n = strtoull(argv[i + 1], 0, 10);
    return 2;
}

// ---- throughput -------------------------------------------------------------

typedef struct {
    uint64_t k, n;
    ahn_opts o;
    double deadline;
    _Atomic int *ready;
    uint64_t iters;
    double secs;
} tp_arg;

static void *tp_worker(void *a) {
    tp_arg *t = a;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    ahn_ctx *ctx = ahn_ctx_new(t->k, t->n, -1, &t->o);
    ahn_num *x = ahn_num_new(ctx);
    ahn_set_u64(ctx, x, 3);
    ahn_square_n(ctx, x, 5);
    atomic_fetch_add(t->ready, 1);
    while (atomic_load(t->ready) > 0) {}  // start together
    double t0 = now();
    uint64_t it = 0;
    while (now() < t->deadline) {
        ahn_square_n(ctx, x, 4);
        it += 4;
    }
    t->secs = now() - t0;
    t->iters = it;
    ahn_num_free(ctx, x);
    ahn_ctx_free(ctx);
    return NULL;
}

// Run `tests` concurrent tests with `threads` threads each; returns squarings/s.
static double throughput(uint64_t k, uint64_t n, ahn_opts o, int tests, int threads, double secs) {
    pthread_t th[64];
    tp_arg a[64];
    _Atomic int ready = 0;
    o.threads = (uint32_t)threads;
    for (int i = 0; i < tests; i++) a[i] = (tp_arg){k, n, o, 0, &ready, 0, 0};
    for (int i = 0; i < tests; i++) pthread_create(&th[i], NULL, tp_worker, &a[i]);
    while (atomic_load(&ready) < tests) {}
    double dl = now() + secs;
    for (int i = 0; i < tests; i++) a[i].deadline = dl;
    atomic_store(&ready, -1);
    double tot = 0;
    for (int i = 0; i < tests; i++) {
        pthread_join(th[i], NULL);
        tot += a[i].iters / a[i].secs;
    }
    return tot;
}

static void progress(void *arg, uint64_t i, uint64_t n, double maxerr) {
    (void)arg;
    static double last;
    double t = now();
    if (t - last < 5) return;
    last = t;
    fprintf(stderr, "\r%5.1f%%  maxerr %.4f", 100.0 * i / n, maxerr);
}

static void usage(const char *p) {
    fprintf(stderr,
            "usage: %s prp K N | \"K*2^N-1\" [--expect HEX] [--base B] [--ckpt FILE]\n"
            "       %s time K N [--iters N]\n"
            "       %s throughput K N [--configs 10x1,5x2,...] [--secs S]\n"
            "       %s tune LMIN LMAX [--out FILE]\n"
            "options: --backend ref|c2p|neon --fft L --threads T\n"
            "         --order 0..3 (FFT stage order, default from the tuning file)\n",
            p, p, p, p);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    const char *cmd = argv[1];
    uint64_t k = 0, n = 0;
    int used = parse_number(argc, argv, 2, &k, &n);
    if (!used) {
        usage(argv[0]);
        return 2;
    }
    ahn_opts o = {0};
    ahn_prp_opts po = {0};
    const char *expect = NULL, *out = NULL, *cfg = "1x1,5x1,10x1,1x5,2x5,1x8,5x2,2x4";
    uint64_t iters = 1000;
    uint32_t base = 3;
    double secs = 3;
    for (int i = 2 + used; i + 1 < argc; i += 2) {
        const char *a = argv[i], *v = argv[i + 1];
        if (!strcmp(a, "--backend")) o.backend = parse_backend(v);
        else if (!strcmp(a, "--fft")) o.fft_len = (uint32_t)atoi(v);
        else if (!strcmp(a, "--threads")) o.threads = (uint32_t)atoi(v);
        else if (!strcmp(a, "--order")) o.plan_order = 1 + (uint32_t)atoi(v);
        else if (!strcmp(a, "--iters")) iters = strtoull(v, 0, 10);
        else if (!strcmp(a, "--expect")) expect = v;
        else if (!strcmp(a, "--base")) base = (uint32_t)atoi(v);
        else if (!strcmp(a, "--ckpt")) po.checkpoint = v;
        else if (!strcmp(a, "--ckpt-secs")) po.checkpoint_secs = atof(v);
        else if (!strcmp(a, "--block")) po.gerbicz_block = (uint32_t)atoi(v);
        else if (!strcmp(a, "--fault")) po.inject_fault_at = strtoull(v, 0, 10);
        else if (!strcmp(a, "--configs")) cfg = v;
        else if (!strcmp(a, "--secs")) secs = atof(v);
        else if (!strcmp(a, "--out")) out = v;
        else {
            fprintf(stderr, "unknown option %s\n", a);
            return 2;
        }
    }

    if (!strcmp(cmd, "prp")) {
        po.fft = o;
        po.progress = progress;
        ahn_prp_result r;
        int rc = ahn_prp(k, n, base, &po, &r);
        fprintf(stderr, "\r                                        \r");
        if (rc) {
            printf("%" PRIu64 "*2^%" PRIu64 "-1  ERROR %d\n", k, n, rc);
            return 1;
        }
        printf("%" PRIu64 "*2^%" PRIu64 "-1 is %s  RES64 %016" PRIX64
               "  FFT %u  %.1f s (%.1f us/iter)  maxerr %.4f  gerbicz_errors %u  fft_retries %u%s\n",
               k, n, r.is_prp ? "a probable prime!" : "composite", r.res64, r.fft_len, r.seconds,
               r.seconds * 1e6 / n, r.maxerr, r.gerbicz_errors, r.roundoff_retries,
               r.resumed ? "  (resumed)" : "");
        if (expect) {
            int ok = strtoull(expect, 0, 16) == r.res64;
            printf("%s expected %s\n", ok ? "MATCH" : "MISMATCH", expect);
            return !ok;
        }
        return 0;
    }

    if (!strcmp(cmd, "time")) {
        ahn_ctx *ctx = ahn_ctx_new(k, n, -1, &o);
        if (!ctx) {
            fprintf(stderr, "cannot create context\n");
            return 1;
        }
        ahn_num *x = ahn_num_new(ctx);
        printf("%" PRIu64 "*2^%" PRIu64 "-1  FFT %u (%.2f bits/word)  backend %s\n", k, n,
               ahn_fft_len(ctx), (double)n / ahn_fft_len(ctx), ahn_backend_name(ctx));
        ahn_set_u64(ctx, x, 3);
        ahn_square_n(ctx, x, 10);
        double t0 = now();
        ahn_square_n(ctx, x, iters);
        double dt = now() - t0;
        printf("%.3f ms/iter  (%.1f us)  maxerr %.4f\n", dt * 1e3 / iters, dt * 1e6 / iters,
               ahn_max_roundoff(ctx));
        ahn_num_free(ctx, x);
        ahn_ctx_free(ctx);
        return 0;
    }

    if (!strcmp(cmd, "throughput")) {
        ahn_ctx *c0 = ahn_ctx_new(k, n, -1, &o);
        if (!c0) {
            fprintf(stderr, "cannot create context\n");
            return 1;
        }
        o.fft_len = ahn_fft_len(c0);
        ahn_ctx_free(c0);
        printf("%" PRIu64 "*2^%" PRIu64 "-1  FFT %u\n", k, n, o.fft_len);
        char buf[256];
        snprintf(buf, sizeof buf, "%s", cfg);
        double best = 0;
        char bestc[32] = "";
        for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
            int T = 1, S = 1;
            if (sscanf(tok, "%dx%d", &T, &S) != 2 || T < 1 || S < 1 || T > 64) continue;
            double r = throughput(k, n, o, T, S, secs);
            printf("%-6s %10.0f sq/s  %8.1f tests/hour  (%.1f us/iter per test)\n", tok, r,
                   r * 3600.0 / n, 1e6 * T / r);
            fflush(stdout);
            if (r > best) {
                best = r;
                snprintf(bestc, sizeof bestc, "%s", tok);
            }
        }
        printf("best: %s  %.1f tests/hour\n", bestc, best * 3600.0 / n);
        return 0;
    }

    if (!strcmp(cmd, "tune")) {
        // time every usable FFT length in [k, n] (here k = LMIN, n = LMAX) in
        // each of the four stage orders, interleaved so that drift in machine
        // load hits them alike; record the fastest order with its time
        FILE *f = out ? fopen(out, "a") : NULL;
        for (uint32_t L = ((uint32_t)k + 63) / 64 * 64; L <= n; L += 64) {
            ahn_ctx *c[4] = {0};
            ahn_num *x[4] = {0};
            double best[4] = {1e30, 1e30, 1e30, 1e30};
            int ok = 1;
            for (int ord = 0; ord < 4 && ok; ord++) {
                ahn_opts lo = o;
                lo.fft_len = L;
                lo.plan_order = 1 + ord;
                // any n with ~12 bits/word and k = 1 exercises the length
                c[ord] = ahn_ctx_new(1, (uint64_t)L * 12, -1, &lo);
                if (!c[ord]) {
                    ok = 0;
                    break;
                }
                x[ord] = ahn_num_new(c[ord]);
                ahn_set_u64(c[ord], x[ord], 3);
                ahn_square_n(c[ord], x[ord], 3);
            }
            for (int rep = 0; ok && rep < 3; rep++) {
                for (int ord = 0; ord < 4; ord++) {
                    uint64_t it = 0;
                    double t0 = now(), t1;
                    do {
                        ahn_square_n(c[ord], x[ord], 2);
                        it += 2;
                    } while ((t1 = now()) - t0 < 0.05);
                    double us = (t1 - t0) * 1e6 / it;
                    if (us < best[ord]) best[ord] = us;
                }
            }
            if (ok) {
                int bo = 0;
                for (int ord = 1; ord < 4; ord++) {
                    if (best[ord] < best[bo]) bo = ord;
                }
                printf("%u %.2f %d\n", L, best[bo], bo);
                if (f) fprintf(f, "%u %.2f %d\n", L, best[bo], bo);
                fflush(stdout);
            }
            for (int ord = 0; ord < 4; ord++) {
                if (x[ord]) ahn_num_free(c[ord], x[ord]);
                if (c[ord]) ahn_ctx_free(c[ord]);
            }
        }
        if (f) fclose(f);
        return 0;
    }

    usage(argv[0]);
    return 2;
}
