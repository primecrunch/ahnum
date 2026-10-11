#ifndef AHN_THREADS_H
#define AHN_THREADS_H

#include <stdint.h>

typedef struct ahn_pool ahn_pool;
typedef void (*ahn_pool_fn)(void *arg, int id, int n);

ahn_pool *ahn_pool_new(int n);  // n workers including the caller
void ahn_pool_free(ahn_pool *p);
int ahn_pool_size(const ahn_pool *p);
// Run fn(arg, id, n) on every worker and wait for all of them.
void ahn_pool_run(ahn_pool *p, ahn_pool_fn fn, void *arg);

#endif
