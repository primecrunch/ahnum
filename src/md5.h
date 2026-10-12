// MD5 (RFC 1321). Used only to reproduce PRST's proof hashes, fingerprints and .md5 files.
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t a, b, c, d;
    uint64_t len;  // bytes
    uint8_t buf[64];
} kite_md5;

void kite_md5_init(kite_md5 *m);
void kite_md5_update(kite_md5 *m, const void *data, size_t len);
void kite_md5_final(kite_md5 *m, uint8_t out[16]);
void kite_md5_hex(const void *data, size_t len, char out[33]);  // lowercase, NUL-terminated
