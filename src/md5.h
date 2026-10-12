// MD5 (RFC 1321). Used only to reproduce PRST's proof hashes, fingerprints and .md5 files.
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t a, b, c, d;
    uint64_t len;  // bytes
    uint8_t buf[64];
} ahn_md5;

void ahn_md5_init(ahn_md5 *m);
void ahn_md5_update(ahn_md5 *m, const void *data, size_t len);
void ahn_md5_final(ahn_md5 *m, uint8_t out[16]);
void ahn_md5_hex(const void *data, size_t len, char out[33]);  // lowercase, NUL-terminated
