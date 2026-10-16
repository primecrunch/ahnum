// LLR error paths, driven through the API with the testing hooks: a corrupted value in one or
// more runs (confirmation and the three-run vote), round-off failures in the doublings and in
// the Lucas ladder, a genuinely too-short FFT, giving up, and resuming from a checkpoint when
// stopped inside run 0 and inside the confirmation run. Numbers and residues are checked
// against PRST 14.0.1888 (see CMakeLists.txt): 3*2^41628-1 is prime, 9357*2^25000-1 is
// composite with RES64 54A27E03D75F2447.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "kite.h"

static const uint64_t PK = 3, PN = 41628;
static const uint64_t CK = 9357, CN = 25000, CRES = 0x54A27E03D75F2447ull;
static const char *CKPT = "test_llr.ckpt";

static int fails = 0;

#define EXPECT(cond, ...)                         \
    do {                                          \
        if (!(cond)) {                            \
            printf("  FAIL: " __VA_ARGS__);       \
            printf("  (%s)\n", #cond);            \
            fails++;                              \
        }                                         \
    } while (0)

// Stops the test once `run` reaches `at` doublings (run is tracked through on_confirm).
typedef struct {
    int run, stop_run;
    uint64_t stop_at;
    volatile int abort;
} stopper;

static void st_progress(void *arg, uint64_t iter, uint64_t total, double maxerr) {
    (void)total, (void)maxerr;
    stopper *s = arg;
    if (s->stop_run >= 0 && s->run == s->stop_run && iter >= s->stop_at) s->abort = 1;
}

static void st_confirm(void *arg, int run) { ((stopper *)arg)->run = run; }

static int run(const char *name, uint64_t k, uint64_t n, kite_llr_opts *o, kite_llr_result *r) {
    int rc = kite_llr(k, n, o, r);
    printf("%-34s rc %d  %s  RES64 %016" PRIX64 "  FFT %u  retries %u  confirm %u  maxerr %.3f\n",
           name, rc, r->is_prime ? "prime    " : "composite", r->res64, r->fft_len,
           r->roundoff_retries, r->confirm_runs, r->maxerr);
    return rc;
}

static void expect_prime(const kite_llr_result *r, int rc) {
    EXPECT(rc == 0, "rc %d\n", rc);
    EXPECT(r->is_prime, "3*2^41628-1 not reported prime\n");
}

static void expect_composite(const kite_llr_result *r, int rc) {
    EXPECT(rc == 0, "rc %d\n", rc);
    EXPECT(!r->is_prime && r->res64 == CRES, "RES64 %016" PRIX64 "\n", r->res64);
}

int main(void) {
    kite_llr_opts o;
    kite_llr_result r;
    int rc;
    remove(CKPT);

    // A prime whose run 0 is wrong: runs 1 and 2 must outvote it.
    o = (kite_llr_opts){0};
    o.inject_fault_at = 20000;
    rc = run("prime, fault in run 0", PK, PN, &o, &r);
    expect_prime(&r, rc);
    EXPECT(r.confirm_runs == 2, "confirm_runs %u\n", r.confirm_runs);

    // A composite with run 0 wrong, then with run 1 wrong: run 2 decides each time.
    o = (kite_llr_opts){0};
    o.inject_fault_at = 12000;
    rc = run("composite, fault in run 0", CK, CN, &o, &r);
    expect_composite(&r, rc);
    EXPECT(r.confirm_runs == 2, "confirm_runs %u\n", r.confirm_runs);

    o.inject_fault_runs = 2;
    rc = run("composite, fault in run 1", CK, CN, &o, &r);
    expect_composite(&r, rc);
    EXPECT(r.confirm_runs == 2, "confirm_runs %u\n", r.confirm_runs);

    // Runs 0 and 1 both wrong (differently, at different FFT lengths): three answers, error.
    o.inject_fault_runs = 3;
    rc = run("composite, faults in runs 0 and 1", CK, CN, &o, &r);
    EXPECT(rc == -3, "rc %d, expected -3\n", rc);

    // A round-off failure in the doublings: redo the block at a longer FFT, same answer.
    o = (kite_llr_opts){0};
    o.inject_roundoff_at = 15000;
    rc = run("prime, round-off fault", PK, PN, &o, &r);
    expect_prime(&r, rc);
    EXPECT(r.roundoff_retries == 1, "retries %u\n", r.roundoff_retries);
    uint32_t L0;
    {
        kite_llr_opts p = {0};
        kite_llr_result q;
        kite_llr(PK, PN, &p, &q);
        L0 = q.fft_len;
    }
    EXPECT(r.fft_len > L0, "FFT %u not longer than %u\n", r.fft_len, L0);

    // The same with a checkpoint, so the redo goes through the file.
    o.checkpoint = CKPT;
    o.checkpoint_secs = 1e-9;
    rc = run("prime, round-off fault, checkpoint", PK, PN, &o, &r);
    expect_prime(&r, rc);
    EXPECT(r.roundoff_retries == 1, "retries %u\n", r.roundoff_retries);

    // A round-off failure in the Lucas ladder, with and without a checkpoint: the bad V_k must
    // not be saved or reused.
    o = (kite_llr_opts){0};
    o.inject_roundoff_ladder = 1;
    rc = run("composite, ladder round-off fault", CK, CN, &o, &r);
    expect_composite(&r, rc);
    EXPECT(r.roundoff_retries == 1, "retries %u\n", r.roundoff_retries);
    o.checkpoint = CKPT;
    o.checkpoint_secs = 1e-9;
    rc = run("same, with checkpoint", CK, CN, &o, &r);
    expect_composite(&r, rc);
    EXPECT(r.roundoff_retries == 1, "retries %u\n", r.roundoff_retries);

    // A genuinely too-short FFT: real round-off failures, then the right answer.
    o = (kite_llr_opts){0};
    o.fft.fft_len = kite_next_fft_len(L0 / 2);
    rc = run("prime, FFT half the default", PK, PN, &o, &r);
    expect_prime(&r, rc);
    EXPECT(r.roundoff_retries >= 1, "retries %u\n", r.roundoff_retries);

    // A round-off limit nothing can meet: give up with -2 rather than loop.
    o = (kite_llr_opts){0};
    o.max_roundoff = 1e-9;
    rc = run("round-off limit 1e-9", PK, PN, &o, &r);
    EXPECT(rc == -2, "rc %d, expected -2\n", rc);

    // Stopped in run 0, resumed; then stopped inside the confirmation run (forced by a fault
    // in run 0) and resumed. Each resume must finish with the right answer.
    stopper s = {0, 0, 10000, 0};
    o = (kite_llr_opts){0};
    o.checkpoint = CKPT;
    o.checkpoint_secs = 1e-9;
    o.progress = st_progress;
    o.on_confirm = st_confirm;
    o.progress_arg = &s;
    o.abort_flag = &s.abort;
    rc = run("composite, stopped in run 0", CK, CN, &o, &r);
    EXPECT(rc == KITE_LLR_ABORTED, "rc %d\n", rc);
    s = (stopper){0, -1, 0, 0};
    rc = run("  resumed", CK, CN, &o, &r);
    expect_composite(&r, rc);
    EXPECT(r.resumed && r.resumed_at >= 10000, "resumed %d at %" PRIu64 "\n", r.resumed,
           r.resumed_at);
    EXPECT(r.confirm_runs == 1, "confirm_runs %u\n", r.confirm_runs);

    s = (stopper){0, 1, 15000, 0};
    o.inject_fault_at = 5000;
    rc = run("prime, fault, stopped in run 1", PK, PN, &o, &r);
    EXPECT(rc == KITE_LLR_ABORTED, "rc %d\n", rc);
    s = (stopper){0, -1, 0, 0};
    rc = run("  resumed", PK, PN, &o, &r);
    expect_prime(&r, rc);
    EXPECT(r.resumed && r.resumed_at >= 15000, "resumed %d at %" PRIu64 "\n", r.resumed,
           r.resumed_at);
    EXPECT(r.confirm_runs == 2, "confirm_runs %u\n", r.confirm_runs);

    remove(CKPT);
    printf(fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails != 0;
}
