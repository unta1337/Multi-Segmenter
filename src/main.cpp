#include "library.h"
#include <cstdio>
#include <omp.h>

int main() {
    printf("Library macro! [%d]\n", LIBRARY_MACRO(1));
    printf("Library func! [%d]\n", func1(1));

    #pragma omp parallel num_threads(10)
    printf("Current thread: %d\n", omp_get_thread_num());

    return 0;
}