// PRST-compatible Fermat proofs for k*2^n-1 (PRST's "smooth" path: x_0 = a^k, then n
// squarings). Produces and reads PRST's point, product and certificate files, so a
// certificate made here verifies with `prst -proof cert` and the other way round.
// File formats: docs/prst-compat.md. The algorithms port PRST 14.1 src/proof.cpp.
#pragma once

#include <stdint.h>

#include "kite.h"

// A non-negative integer as PRST stores it: little-endian 32-bit words, no leading zeros.
typedef struct {
    uint32_t *w;
    int32_t len;
} pp_giant;

void pp_giant_free(pp_giant *g);

enum { PP_TYPE_VALUE = 1, PP_TYPE_PRODUCT = 3, PP_TYPE_CERT = 4 };

uint32_t pp_fingerprint(uint64_t k, uint64_t n);             // N mod 3417905339
uint32_t pp_unique_fingerprint(uint32_t fp, const char *id);  // first word of MD5(fp || id)
// Fingerprint inside point and product files for base a: unique(fp, "a.n").
uint32_t pp_proof_fingerprint(uint64_t k, uint64_t n, uint32_t a);

// Write a state file and its .md5 sidecar. Returns 0 on success.
int pp_write(const char *path, int type, uint32_t fingerprint, int32_t iteration,
             const pp_giant *g, int ng);
// Read a state file, checking magic, type, fingerprint and the .md5 sidecar if present.
// Fills up to maxg giants (missing trailing giants are left empty). Returns 0 on success.
int pp_read(const char *path, int type, uint32_t fingerprint, int32_t *iteration, pp_giant *g,
            int maxg);

// Positions of the count+1 proof points along `iterations` squarings (count a power of 2).
// Returns PRST's M (the spacing it reports).
int pp_points(uint32_t iterations, int count, uint32_t *pos);

// Conversions between a kite number and a giant (value in [0, N)).
void pp_from_num(kite_ctx *ctx, const kite_num *x, pp_giant *g);
void pp_to_num(kite_ctx *ctx, const pp_giant *g, kite_num *x);
uint64_t pp_res64(const pp_giant *g);

// `prefix` is "prst_<fp>" in the working directory.
// Save step: points 0..count are on disk; writes the products and returns the raw
// certificate RES64. Deletes points 1..count-1 unless keep.
int pp_save_products(kite_ctx *ctx, uint64_t k, uint64_t n, uint32_t a, int count,
                     const char *prefix, int keep, uint64_t *raw_res64);
// Build step: checks point 0 = a^k, writes prefix.cert, returns the certificate RES64.
// Deletes points 0 and count and the products unless keep.
int pp_build_cert(kite_ctx *ctx, uint64_t k, uint64_t n, uint32_t a, int count,
                  const char *prefix, int keep, uint64_t *cert_res64);
// Verify: reads a certificate, squares its value M times with a Gerbicz check and returns
// the RES64 PRST prints as "certificate RES64".
int pp_verify_cert(kite_ctx *ctx, uint64_t k, uint64_t n, const char *path, uint64_t *cert_res64,
                   uint32_t *M);
