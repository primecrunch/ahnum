// MD5 (RFC 1321).
#include "md5.h"

#include <string.h>

static const uint32_t K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
static const uint8_t S[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                              5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                              4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                              6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

static void block(kite_md5 *m, const uint8_t *p) {
    uint32_t w[16];
    for (int i = 0; i < 16; i++)
        w[i] = p[4 * i] | p[4 * i + 1] << 8 | p[4 * i + 2] << 16 | (uint32_t)p[4 * i + 3] << 24;
    uint32_t a = m->a, b = m->b, c = m->c, d = m->d;
    for (int i = 0; i < 64; i++) {
        uint32_t f;
        int g;
        if (i < 16) {
            f = (b & c) | (~b & d);
            g = i;
        } else if (i < 32) {
            f = (d & b) | (~d & c);
            g = (5 * i + 1) & 15;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3 * i + 5) & 15;
        } else {
            f = c ^ (b | ~d);
            g = (7 * i) & 15;
        }
        uint32_t t = a + f + K[i] + w[g];
        a = d;
        d = c;
        c = b;
        b += t << S[i] | t >> (32 - S[i]);
    }
    m->a += a;
    m->b += b;
    m->c += c;
    m->d += d;
}

void kite_md5_init(kite_md5 *m) {
    m->a = 0x67452301;
    m->b = 0xefcdab89;
    m->c = 0x98badcfe;
    m->d = 0x10325476;
    m->len = 0;
}

void kite_md5_update(kite_md5 *m, const void *data, size_t len) {
    const uint8_t *p = data;
    size_t have = m->len & 63;
    m->len += len;
    if (have) {
        size_t take = 64 - have < len ? 64 - have : len;
        memcpy(m->buf + have, p, take);
        p += take;
        len -= take;
        if (have + take < 64) return;
        block(m, m->buf);
    }
    for (; len >= 64; p += 64, len -= 64) block(m, p);
    memcpy(m->buf, p, len);
}

void kite_md5_final(kite_md5 *m, uint8_t out[16]) {
    uint64_t bits = m->len * 8;
    uint8_t pad[72] = {0x80};
    size_t have = m->len & 63, n = (have < 56 ? 56 : 120) - have;
    kite_md5_update(m, pad, n);
    for (int i = 0; i < 8; i++) pad[i] = (uint8_t)(bits >> (8 * i));
    kite_md5_update(m, pad, 8);
    uint32_t r[4] = {m->a, m->b, m->c, m->d};
    for (int i = 0; i < 16; i++) out[i] = (uint8_t)(r[i / 4] >> (8 * (i % 4)));
}

void kite_md5_hex(const void *data, size_t len, char out[33]) {
    kite_md5 m;
    uint8_t d[16];
    kite_md5_init(&m);
    kite_md5_update(&m, data, len);
    kite_md5_final(&m, d);
    for (int i = 0; i < 16; i++) {
        out[2 * i] = "0123456789abcdef"[d[i] >> 4];
        out[2 * i + 1] = "0123456789abcdef"[d[i] & 15];
    }
    out[32] = 0;
}
