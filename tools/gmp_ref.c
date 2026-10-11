// GMP oracle: Fermat PRP residues for N = k*2^n + c (default c = -1).
// Usage: gmp_ref k n [base] [c] [--all]
// Prints the low 64 bits of base^(N-1) mod N as upper-case hex.
// --all also prints candidate alternative conventions (used to pin down the
// PrimeCrunch res64 definition).
#include <gmp.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t low64(const mpz_t x) {
    uint64_t r = 0;
    size_t cnt = 0;
    uint64_t buf[1] = {0};
    if (mpz_sizeinbase(x, 2) <= 64) {
        mpz_export(buf, &cnt, -1, sizeof(uint64_t), 0, 0, x);
        return buf[0];
    }
    mpz_t t;
    mpz_init(t);
    mpz_fdiv_r_2exp(t, x, 64);
    mpz_export(&r, &cnt, -1, sizeof(uint64_t), 0, 0, t);
    mpz_clear(t);
    return r;
}

static void show(const char *label, const mpz_t x) {
    printf("%-28s %016" PRIX64 "\n", label, low64(x));
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s k n [base] [c] [--all]\n", argv[0]);
        return 2;
    }
    unsigned long k = strtoul(argv[1], 0, 10);
    unsigned long n = strtoul(argv[2], 0, 10);
    unsigned long base = argc > 3 && argv[3][0] != '-' ? strtoul(argv[3], 0, 10) : 3;
    long c = argc > 4 && strcmp(argv[4], "--all") ? strtol(argv[4], 0, 10) : -1;
    int all = !strcmp(argv[argc - 1], "--all");

    mpz_t N, e, b, r;
    mpz_inits(N, e, b, r, NULL);
    mpz_set_ui(N, k);
    mpz_mul_2exp(N, N, n);
    if (c < 0) mpz_sub_ui(N, N, (unsigned long)-c);
    else mpz_add_ui(N, N, (unsigned long)c);
    mpz_set_ui(b, base);

    mpz_sub_ui(e, N, 1);
    mpz_powm(r, b, e, N);
    if (!all) {
        printf("%016" PRIX64 "\n", low64(r));
    } else {
        show("b^(N-1) mod N", r);
        // b^(k*2^n) mod N  (= b^(N+1) for c=-1)
        mpz_set_ui(e, k);
        mpz_mul_2exp(e, e, n);
        mpz_powm(r, b, e, N);
        show("b^(k*2^n) mod N", r);
        mpz_add_ui(e, N, 1);
        mpz_powm(r, b, e, N);
        show("b^(N+1) mod N", r);
        mpz_sub_ui(e, N, 1);
        mpz_fdiv_q_2exp(e, e, 1);
        mpz_powm(r, b, e, N);
        show("b^((N-1)/2) mod N", r);
        mpz_add_ui(e, N, 1);
        mpz_fdiv_q_2exp(e, e, 1);
        mpz_powm(r, b, e, N);
        show("b^((N+1)/2) mod N", r);
        // b^(k*2^(n-1)) etc.
        mpz_set_ui(e, k);
        mpz_mul_2exp(e, e, n - 1);
        mpz_powm(r, b, e, N);
        show("b^(k*2^(n-1)) mod N", r);
        mpz_set_ui(e, k);
        mpz_mul_2exp(e, e, n - 2);
        mpz_powm(r, b, e, N);
        show("b^(k*2^(n-2)) mod N", r);
    }
    mpz_clears(N, e, b, r, NULL);
    return 0;
}
