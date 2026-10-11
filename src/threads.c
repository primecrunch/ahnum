// Minimal fork-join thread pool. The caller participates as worker 0; the
// other workers spin briefly on a generation counter and then sleep on a
// condition variable, so idle pools do not burn CPU.
#include "threads.h"

#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdlib.h>

struct ahn_pool {
    int n;
    pthread_t *th;
    _Atomic uint64_t gen;
    _Atomic int remaining;
    _Atomic int quit;
    ahn_pool_fn fn;
    void *arg;
    pthread_mutex_t mu;
    pthread_cond_t cv;
};

typedef struct {
    ahn_pool *p;
    int id;
} worker_arg;

static inline uint64_t ticks(void) {
    uint64_t t;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t));
    return t;
}

static inline uint64_t tick_freq(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}

static void *worker(void *a) {
    worker_arg *wa = a;
    ahn_pool *p = wa->p;
    int id = wa->id;
    free(wa);
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint64_t seen = 0, spin_ticks = tick_freq() / 200;  // spin ~5 ms before sleeping
    for (;;) {
        uint64_t g, t0 = ticks();
        while ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&p->quit, memory_order_relaxed)) return NULL;
            if (ticks() - t0 < spin_ticks) {
                __asm__ volatile("yield");
                continue;
            }
            pthread_mutex_lock(&p->mu);
            while (atomic_load(&p->gen) == seen && !atomic_load(&p->quit))
                pthread_cond_wait(&p->cv, &p->mu);
            pthread_mutex_unlock(&p->mu);
            t0 = ticks();
        }
        seen = g;
        p->fn(p->arg, id, p->n);
        atomic_fetch_sub_explicit(&p->remaining, 1, memory_order_acq_rel);
    }
}

ahn_pool *ahn_pool_new(int n) {
    if (n < 1) n = 1;
    ahn_pool *p = calloc(1, sizeof *p);
    p->n = n;
    if (n > 1) pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    p->th = calloc((size_t)n, sizeof(pthread_t));
    for (int i = 1; i < n; i++) {
        worker_arg *wa = malloc(sizeof *wa);
        wa->p = p;
        wa->id = i;
        pthread_create(&p->th[i], NULL, worker, wa);
    }
    return p;
}

void ahn_pool_free(ahn_pool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    atomic_store(&p->quit, 1);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->th[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    free(p->th);
    free(p);
}

int ahn_pool_size(const ahn_pool *p) { return p ? p->n : 1; }

void ahn_pool_run(ahn_pool *p, ahn_pool_fn fn, void *arg) {
    if (!p || p->n == 1) {
        fn(arg, 0, 1);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    atomic_store_explicit(&p->remaining, p->n - 1, memory_order_relaxed);
    pthread_mutex_lock(&p->mu);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    fn(arg, 0, p->n);
    while (atomic_load_explicit(&p->remaining, memory_order_acquire) != 0) __asm__ volatile("yield");
}
