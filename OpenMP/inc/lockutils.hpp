#ifndef __LOCKUTILS_H
#define __LOCKUTILS_H

#define LOCK(locks, index, expression)                                                                                 \
    omp_set_lock(&locks[index]);                                                                                       \
    expression;                                                                                                        \
    omp_unset_lock(&locks[index])

#include <omp.h>

inline omp_lock_t* new_locks(size_t size) {
    omp_lock_t* locks = new omp_lock_t[size];
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        omp_init_lock(&locks[i]);
    }
    return locks;
}

inline void destroy_locks(omp_lock_t* locks, size_t size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        omp_destroy_lock(&locks[i]);
    }
    delete locks;
}

#endif
