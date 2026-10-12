#ifndef KITE_THREADS_H
#define KITE_THREADS_H

#include <stdint.h>

typedef struct kite_pool kite_pool;
typedef void (*kite_pool_fn)(void *arg, int id, int n);

kite_pool *kite_pool_new(int n);  // n workers including the caller
void kite_pool_free(kite_pool *p);
int kite_pool_size(const kite_pool *p);
// Run fn(arg, id, n) on every worker and wait for all of them.
void kite_pool_run(kite_pool *p, kite_pool_fn fn, void *arg);

#endif
