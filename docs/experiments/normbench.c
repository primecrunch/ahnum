// Time kite_norm_ao / _aoe on L1-resident data and table (one group, R = 96, rowstride 4224).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mach/mach_time.h>
typedef double (*norm_fn)(double *, size_t, size_t, double *, const double *, const double *,
                          const uint64_t *, const double *);
extern double kite_norm_ao(double *, size_t, size_t, double *, const double *, const double *, const uint64_t *, const double *);
extern double kite_norm_aoe(double *, size_t, size_t, double *, const double *, const double *, const uint64_t *, const double *);
int main(int argc, char **argv) {
    size_t R = 96, rs = 4224, np = R / 2, reps = argc > 1 ? atol(argv[1]) : 200000;
    double *d = aligned_alloc(128, R * rs), *tab = aligned_alloc(128, np * 768), carr[96] = {0};
    for (size_t r = 0; r < R; r++) for (int i = 0; i < 16; i++) d[r * rs / 8 + i] = (double)((r * 7 + i) % 5000);
    for (size_t k = 0; k < np * 96; k += 6) { tab[k] = tab[k+1] = 1.0; tab[k+2] = tab[k+3] = 1.0 / 8192; tab[k+4] = tab[k+5] = 8192; }
    norm_fn fns[2] = {kite_norm_ao, kite_norm_aoe}; const char *nm[2] = {"ao", "aoe"};
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    for (int round = 0; round < 3; round++)
    for (int f = 0; f < 2; f++) {
        for (size_t i = 0; i < 1000; i++) fns[f](d, rs, np, carr, tab, 0, 0, 0);
        uint64_t t0 = mach_absolute_time();
        for (size_t i = 0; i < reps; i++) fns[f](d, rs, np, carr, tab, 0, 0, 0);
        uint64_t t1 = mach_absolute_time();
        double ns = (double)(t1 - t0) * tb.numer / tb.denom / reps / (np / 4);  // per 4-pair iteration
        printf("%-4s %.2f ns per iteration (128 digits)  -> %.2f us per squaring (384 its)\n", nm[f], ns, ns * 384 / 1000);
    }
    return 0;
}
