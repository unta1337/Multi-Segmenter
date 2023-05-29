#include "matrixutil.h"

KERNEL_DEF(__global__, matrix_print,
    size_t rows = *(size_t*)argv[0];
    size_t cols = *(size_t*)argv[1];
    size_t stride = *(size_t*)argv[2];
    int* elems = (int*)argv[3];

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++)
            printf("%d ", elems[i * stride + j]);
        printf("\n");
    }
)
