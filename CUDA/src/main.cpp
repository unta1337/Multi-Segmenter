#include <stdio.h>

#include "matrixutil.h"

int main() {
    size_t rows = 3;
    size_t cols = 4;
    size_t stride = 4;
    int* elems = new int[rows * cols];
    for (size_t i = 0; i < rows * cols; i++)
        elems[i] = i;

    void* argv[] = {
        to_dev_new(&rows, sizeof(rows)),
        to_dev_new(&cols, sizeof(cols)),
        to_dev_new(&stride, sizeof(stride)),
        to_dev_new(elems, rows * cols * sizeof(*elems)),
    };
    matrix_print_k(dim3(1, 1, 1), dim3(1, 1, 1), sizeof(argv) / sizeof(argv[0]), (void**)argv);

    return 0;
}
