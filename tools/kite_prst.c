// kite-prst: a slice of PRST's command line (Fermat tests and proofs of k*2^n-1), on kite.
// See docs/prst-compat.md. Anything outside that slice (other forms and options) goes to PRST:
// $KITE_PRST, or `prst` next to this binary.
#include <inttypes.h>
#include <libgen.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "kite.h"
#include "prstproof.h"

#define PROGRESS_BLANK "                                                                      "

static volatile int abort_flag = 0;
static int progress_pending = 0;  // a progress reading is on screen without a line end

static void on_signal(int sig) {
    (void)sig;
    abort_flag = 1;
}

static double now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Print a full line, first clearing a progress reading the way PRST does.
static void out(const char *fmt, ...) {
    if (progress_pending) {
        fputs(PROGRESS_BLANK "\r", stdout);
        progress_pending = 0;
    }
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
}

static void result_save(const char *fmt, ...) {
    FILE *f = fopen("result.txt", "a");
    if (!f) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fclose(f);
}

static void hand_to_prst(char **argv) {
    const char *env = getenv("KITE_PRST");
    char path[4096];
    if (env && *env) {
        snprintf(path, sizeof path, "%s", env);
    } else {
        char self[4096];
        snprintf(self, sizeof self, "%s", argv[0]);
        snprintf(path, sizeof path, "%s/prst", dirname(self));
    }
    argv[0] = path;
    execv(path, argv);
    fprintf(stderr, "kite-prst: cannot run PRST at %s for this test\n", path);
    exit(2);
}

typedef struct {
    char expr[128];
    uint64_t k, n;
    uint32_t a;
    int threads;
    double progress_secs;
    double write_secs;  // unverified checkpoint interval
    int fermat, fft_info, version, llr;
    int proof_op;  // 0 none, 1 save, 2 build, 3 cert
    int proof_count, proof_keep;
    const char *cert_path;
} args;

static int parse_expr(args *A, const char *s) {
    unsigned long long k, n;
    int used = 0;
    if (sscanf(s, "%llu*2^%llu-1%n", &k, &n, &used) != 2 || s[used] != 0) return -1;
    if (k == 0 || k % 2 == 0 || n < 2) return -1;
    snprintf(A->expr, sizeof A->expr, "%s", s);
    A->k = k;
    A->n = n;
    return 0;
}

// Returns 0 if handled by kite-prst, 1 to hand to PRST, -1 on a usage error.
static int parse(args *A, int argc, char **argv) {
    memset(A, 0, sizeof *A);
    A->a = 3;
    A->threads = 1;
    A->progress_secs = 10;
    A->write_secs = 60;
    int have_expr = 0;
    for (int i = 1; i < argc; i++) {
        const char *s = argv[i];
#define NEXT (i + 1 < argc ? argv[++i] : "")
        if (!strcmp(s, "-v")) {
            A->version = 1;
        } else if (!strcmp(s, "-d") || !strcmp(s, "-q")) {
        } else if (!strcmp(s, "-t")) {
            A->threads = atoi(NEXT);
        } else if (!strcmp(s, "-cpu")) {
            (void)NEXT;  // kite picks its own kernels
        } else if (!strcmp(s, "-fermat")) {
            A->fermat = 1;
            if (i + 2 < argc && !strcmp(argv[i + 1], "a")) {
                i++;
                A->a = (uint32_t)strtoul(NEXT, NULL, 10);
            }
        } else if (!strcmp(s, "-time")) {
            // PRST's -time [write <sec>] [progress <sec>] [coarse]; write sets how often the
            // unverified checkpoint is saved
            int any = 0;
            for (;;) {
                if (i + 2 < argc && !strcmp(argv[i + 1], "write")) {
                    i++;
                    A->write_secs = atof(NEXT);
                } else if (i + 2 < argc && !strcmp(argv[i + 1], "progress")) {
                    i++;
                    A->progress_secs = atof(NEXT);
                } else if (i + 1 < argc && !strcmp(argv[i + 1], "coarse")) {
                    i++;
                } else {
                    break;
                }
                any = 1;
            }
            if (!any) return 1;
        } else if (!strcmp(s, "-fft")) {
            if (!strcmp(NEXT, "info")) A->fft_info = 1;
            else return 1;
        } else if (!strcmp(s, "-proof")) {
            const char *op = NEXT;
            if (!strcmp(op, "save") || !strcmp(op, "build")) {
                A->proof_op = op[0] == 's' ? 1 : 2;
                A->proof_count = atoi(NEXT);
                if (A->proof_count < 2 || (A->proof_count & (A->proof_count - 1))) {
                    fprintf(stderr, "proof count is not a power of 2.\n");
                    return -1;
                }
                if (i + 1 < argc && !strcmp(argv[i + 1], "keep")) {
                    A->proof_keep = 1;
                    i++;
                }
            } else if (!strcmp(op, "cert")) {
                A->proof_op = 3;
                A->cert_path = NEXT;
            } else {
                return 1;
            }
        } else if (s[0] == '-') {
            return 1;  // an option outside the supported slice
        } else if (!have_expr) {
            if (parse_expr(A, s)) return 1;
            have_expr = 1;
        } else {
            return 1;
        }
#undef NEXT
    }
    if (A->version) return 0;
    if (!have_expr) return -1;
    if (A->proof_op == 3) return 0;
    if (!A->fermat) {  // PRST's default test for k*2^n-1: Morrison (LLR)
        if (A->fft_info || A->proof_op) return 1;
        A->llr = 1;
        return 0;
    }
    if (A->a < 2) return -1;
    return 0;
}

static void fft_label(uint32_t L, char *buf, size_t len) {
    if (L % (1u << 20) == 0) snprintf(buf, len, "%uM", L >> 20);
    else if (L % 1024 == 0) snprintf(buf, len, "%uK", L >> 10);
    else snprintf(buf, len, "%u", L);
}

static kite_ctx *open_ctx(const args *A, kite_opts *o) {
    memset(o, 0, sizeof *o);
    o->threads = A->threads > 1 ? (uint32_t)A->threads : 1;
    return kite_ctx_new(A->k, A->n, -1, o);
}

static void announce_fft(const args *A, uint32_t L) {
    char lab[32];
    fft_label(L, lab, sizeof lab);
    (void)A;
    out("Using NEON FFT length %s.\n", lab);
}

// ---- Fermat test with optional proof points -------------------------------------------

typedef struct {
    const args *A;
    double t0, last;
    uint64_t last_iter;
    char prefix[64];
    uint32_t fpP;
    uint32_t *pos;
    size_t nlimbs;
    int run;  // LLR: 0, or 1-2 while confirming a composite result
} run_state;

static void show_progress(run_state *R, uint64_t iter, double pct) {
    double t = now();
    if (t - R->last < R->A->progress_secs) return;
    double per_op = iter > R->last_iter ? (t - R->last) * 1e3 / (double)(iter - R->last_iter) : 0;
    if (progress_pending) fputs(PROGRESS_BLANK "\r", stdout);
    printf("%s %.1f%% done, time per op: %.3f ms. \r", R->A->expr, pct, per_op);
    fflush(stdout);
    progress_pending = 1;
    R->last = t;
    R->last_iter = iter;
}

static void progress(void *arg, uint64_t iter, uint64_t n, double maxerr) {
    (void)maxerr;
    show_progress(arg, iter, 100.0 * (double)iter / (double)n);
}

static int write_point(void *arg, uint32_t index, uint64_t iter, const uint64_t *limbs) {
    run_state *R = arg;
    // a full-length value; trimmed to PRST's giant form
    size_t nl = R->nlimbs;
    pp_giant g;
    g.w = malloc(nl * 8);
    for (size_t i = 0; i < nl; i++) {
        g.w[2 * i] = (uint32_t)limbs[i];
        g.w[2 * i + 1] = (uint32_t)(limbs[i] >> 32);
    }
    g.len = (int32_t)(2 * nl);
    while (g.len > 0 && g.w[g.len - 1] == 0) g.len--;
    char path[128];
    snprintf(path, sizeof path, "%s.proof.%u", R->prefix, index);
    int rc = pp_write(path, PP_TYPE_VALUE, R->fpP, (int32_t)iter, &g, 1);
    free(g.w);
    if (rc) out("Write to %s failed.\n", path);
    return rc;
}

static int run_fermat(const args *A) {
    kite_opts o;
    kite_ctx *ctx = open_ctx(A, &o);
    if (!ctx) {
        fprintf(stderr, "kite-prst: cannot set up arithmetic for %s\n", A->expr);
        return 1;
    }
    uint32_t L = kite_fft_len(ctx);
    uint32_t B = (uint32_t)sqrt((double)A->n);
    if (B < 32) B = 32;
    if (B > 2000) B = 2000;
    uint32_t Mb = B / 4 > 8 ? B / 4 : 8;
    uint64_t complexity = A->n + A->n / B;
    announce_fft(A, L);
    out("Fermat probabilistic test of %s, a = %u, complexity = %" PRIu64 ".\n", A->expr, A->a,
        complexity);
    if (A->fft_info) {
        kite_ctx_free(ctx);
        return 0;
    }
    // a divides N: PRST reports the factor (fermat.cpp:350)
    {
        uint64_t r = 1, b = 2 % A->a;
        for (uint64_t e = A->n; e; e >>= 1, b = b * b % A->a)
            if (e & 1) r = r * b % A->a;
        if ((A->k % A->a * r + A->a - 1) % A->a == 0) {
            out("%s is not prime, divisible by %u.\n", A->expr, A->a);
            result_save("%s is not prime, divisible by %u.\n", A->expr, A->a);
            kite_ctx_free(ctx);
            return 0;
        }
    }
    out("%s Gerbicz check enabled, L2 = %u*%u.\n", A->expr, B, Mb);

    run_state R = {.A = A, .nlimbs = kite_limbs(ctx)};
    uint32_t fpN = pp_fingerprint(A->k, A->n);
    snprintf(R.prefix, sizeof R.prefix, "prst_%u", fpN);
    char ckpt[96], ckpt_u[96];
    snprintf(ckpt, sizeof ckpt, "kite_%u.%u.ckpt", fpN, A->a);
    snprintf(ckpt_u, sizeof ckpt_u, "kite_%u.%u.u.ckpt", fpN, A->a);
    kite_prp_opts po = {0};
    po.fft = o;
    po.checkpoint = ckpt;
    po.checkpoint_secs = 300;
    po.checkpoint_unverified = ckpt_u;
    po.unverified_secs = A->write_secs;
    po.progress = progress;
    po.progress_arg = &R;
    po.abort_flag = &abort_flag;
    if (A->proof_op == 1) {
        if (A->n > INT32_MAX) {
            fprintf(stderr, "kite-prst: n too large for PRST proof files\n");
            return 1;
        }
        R.fpP = pp_proof_fingerprint(A->k, A->n, A->a);
        R.pos = malloc((A->proof_count + 1) * sizeof *R.pos);
        pp_points((uint32_t)A->n, A->proof_count, R.pos);
        po.points = R.pos;
        po.npoints = (uint32_t)A->proof_count + 1;
        po.on_point = write_point;
        po.point_arg = &R;
        out("%s Saving %d proof points.\n", A->expr, A->proof_count);
    }
    kite_ctx_free(ctx);

    {  // announce a resume before the test starts, as PRST does; the later of the two files
        uint64_t at = 0, u = 0;
        if (kite_prp_checkpoint_iter(ckpt, A->k, A->n, A->a, &at)) at = 0;
        if (!kite_prp_checkpoint_iter(ckpt_u, A->k, A->n, A->a, &u) && u > at) at = u;
        if (at > 0) out("%s restarting at %.1f%%.\n", A->expr, 100.0 * (double)at / (double)A->n);
    }
    R.t0 = R.last = now();
    kite_prp_result res;
    int rc = kite_prp(A->k, A->n, A->a, &po, &res);
    if (rc == KITE_PRP_ABORTED) {
        out("Terminating...\n");
        free(R.pos);
        return 1;
    }
    if (rc) {
        out("%s test failed (error %d).\n", A->expr, rc);
        free(R.pos);
        return 1;
    }
    remove(ckpt);
    remove(ckpt_u);
    char ckpt_md5[128];
    snprintf(ckpt_md5, sizeof ckpt_md5, "%s.md5", ckpt);
    remove(ckpt_md5);
    double secs = res.seconds;
    if (res.is_prp) {
        out("%s is a probable prime. Time: %.1f s.\n", A->expr, secs);
        result_save("%s is a probable prime. Time: %d s.\n", A->expr, (int)secs);
    } else {
        out("%s is not prime. RES64: %016" PRIX64 ", time: %.1f s.\n", A->expr, res.res64, secs);
        result_save("%s is not prime. RES64: %016" PRIX64 ", time: %d s.\n", A->expr, res.res64,
                    (int)secs);
    }
    if (A->proof_op == 1) {
        double t = now();
        kite_ctx *c2 = open_ctx(A, &o);
        uint64_t raw = 0;
        int prc = c2 ? pp_save_products(c2, A->k, A->n, A->a, A->proof_count, R.prefix,
                                        A->proof_keep, &raw)
                     : -1;
        if (c2) kite_ctx_free(c2);
        free(R.pos);
        if (prc) return 1;
        int depth = 0;
        while ((1 << depth) < A->proof_count) depth++;
        out("%s compressed %d points to %d products, time: %.1f s.\n", A->expr, A->proof_count,
            depth, now() - t);
        out("%s raw certificate RES64: %016" PRIX64 ".\n", A->expr, raw);
        result_save("%s raw certificate RES64: %016" PRIX64 ", time: %d s.\n", A->expr, raw,
                    (int)(now() - t));
    }
    return 0;
}

// ---- certificate build and check ---------------------------------------------------------

// ---- LLR (Morrison) test ----------------------------------------------------------------

// PRST's progress only goes forward, and callers such as the primecrunch client expect that.
// The first run counts 0-99%; a confirmation run holds at 99% until the result line.
#define LLR_FIRST_RUN_PCT 99.0

static double llr_pct(int run, uint64_t iter, uint64_t total) {
    return run ? LLR_FIRST_RUN_PCT : LLR_FIRST_RUN_PCT * (double)iter / (double)total;
}

static void llr_progress(void *arg, uint64_t iter, uint64_t total, double maxerr) {
    (void)maxerr;
    run_state *R = arg;
    show_progress(R, iter, llr_pct(R->run, iter, total));
}

static void llr_confirm(void *arg, int run) {
    run_state *R = arg;
    R->run = run;
    R->last_iter = 0;  // iterations count from 0 again in the new run
    fprintf(stderr, "%s result is composite; confirming with run %d at a longer FFT length.\n",
            R->A->expr, run + 1);
}

static int run_llr(const args *A) {
    uint32_t P;
    uint64_t factor;
    if (kite_llr_choose_P(A->k, A->n, &P, &factor)) {
        fprintf(stderr, "kite-prst: no P found for %s\n", A->expr);
        return 1;
    }
    kite_opts o;
    kite_ctx *ctx = open_ctx(A, &o);
    if (!ctx) {
        fprintf(stderr, "kite-prst: cannot set up arithmetic for %s\n", A->expr);
        return 1;
    }
    announce_fft(A, kite_fft_len(ctx));
    kite_ctx_free(ctx);
    if (factor) {  // PRST reports this before announcing the test (morrison.cpp:200)
        out("%s is not prime, divisible by %" PRIu64 ". Time: 0.0 s.\n", A->expr, factor);
        result_save("%s is not prime, divisible by %" PRIu64 ". Time: 0 s.\n", A->expr, factor);
        return 0;
    }
    out("Morrison (LLR) test of %s, P = %u, Q = -1, complexity = %" PRIu64 ".\n", A->expr, P,
        A->n);
    char ckpt[96];
    snprintf(ckpt, sizeof ckpt, "kite_%u.llr.ckpt", pp_fingerprint(A->k, A->n));
    uint64_t at;
    int at_run = 0;
    if (kite_llr_checkpoint_iter(ckpt, A->k, A->n, &at, &at_run) == 0 && (at > 0 || at_run > 0))
        out("%s restarting at %.1f%%.\n", A->expr, llr_pct(at_run, at, A->n - 1));
    run_state R = {.A = A};
    R.t0 = R.last = now();
    kite_llr_opts lo = {0};
    lo.fft = o;
    lo.checkpoint = ckpt;
    lo.checkpoint_secs = 300;
    lo.progress = llr_progress;
    lo.on_confirm = llr_confirm;
    lo.progress_arg = &R;
    lo.abort_flag = &abort_flag;
    kite_llr_result r;
    int rc = kite_llr(A->k, A->n, &lo, &r);
    if (rc == KITE_LLR_ABORTED) {
        out("Terminating...\n");
        return 1;
    }
    if (rc) {
        out("%s LLR test failed (error %d).\n", A->expr, rc);
        return 1;
    }
    if (r.is_prime) {
        out("%s is prime! Time: %.1f s.\n", A->expr, r.seconds);
        result_save("%s is prime! Time: %d s.\n", A->expr, (int)r.seconds);
    } else {
        out("%s is not a probable prime. Have you run Fermat test first? RES64: %016" PRIX64
            ", time: %.1f s.\n",
            A->expr, r.res64, r.seconds);
        result_save("%s is not a probable prime. Have you run Fermat test first? RES64: %016" PRIX64
                    ", time: %d s.\n",
                    A->expr, r.res64, (int)r.seconds);
    }
    return 0;
}

static int run_build(const args *A) {
    kite_opts o;
    kite_ctx *ctx = open_ctx(A, &o);
    if (!ctx) return 1;
    uint32_t fpN = pp_fingerprint(A->k, A->n);
    char prefix[64];
    snprintf(prefix, sizeof prefix, "prst_%u", fpN);
    double t = now();
    announce_fft(A, kite_fft_len(ctx));
    int depth = 0;
    while ((1 << depth) < A->proof_count) depth++;
    out("%s Building certificate from %d products.\n", A->expr, depth);
    uint64_t res = 0;
    int rc = pp_build_cert(ctx, A->k, A->n, A->a, A->proof_count, prefix, A->proof_keep, &res);
    kite_ctx_free(ctx);
    if (rc) return 1;
    out("%s certificate RES64: %016" PRIX64 ", time: %.1f s.\n", A->expr, res, now() - t);
    result_save("%s certificate RES64: %016" PRIX64 ", time: %d s.\n", A->expr, res,
                (int)(now() - t));
    return 0;
}

static int run_cert(const args *A) {
    kite_opts o;
    kite_ctx *ctx = open_ctx(A, &o);
    if (!ctx) return 1;
    char path[4096];
    if (!A->cert_path || !strcmp(A->cert_path, "default"))
        snprintf(path, sizeof path, "prst_%u.cert", pp_fingerprint(A->k, A->n));
    else
        snprintf(path, sizeof path, "%s", A->cert_path);
    double t = now();
    announce_fft(A, kite_fft_len(ctx));
    uint64_t res = 0;
    uint32_t M = 0;
    int rc = pp_verify_cert(ctx, A->k, A->n, path, &res, &M);
    kite_ctx_free(ctx);
    if (rc) return 1;
    out("Verifying certificate of %s, complexity = %u.\n", A->expr, M);
    out("%s certificate RES64: %016" PRIX64 ", time: %.1f s.\n", A->expr, res, now() - t);
    result_save("%s certificate RES64: %016" PRIX64 ", time: %d s.\n", A->expr, res,
                (int)(now() - t));
    return 0;
}

int main(int argc, char **argv) {
    args A;
    int p = parse(&A, argc, argv);
    if (p == 1) hand_to_prst(argv);
    if (p < 0) {
        fprintf(stderr, "usage: %s K*2^N-1 -d -fermat a 3 [-t T] [-proof save|build C [keep]]\n"
                        "       %s K*2^N-1 -d -proof cert FILE\n       %s -v\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    if (A.version) {
        printf("kite version %s\n", KITE_VERSION);
        return 0;
    }
    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    if (A.proof_op == 2) return run_build(&A);
    if (A.proof_op == 3) return run_cert(&A);
    if (A.llr) return run_llr(&A);
    return run_fermat(&A);
}
