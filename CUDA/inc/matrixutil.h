#ifndef __MATRIXUTIL_H
#define __MATRIXUTIL_H

#include <stdio.h>

#include "kernelutil.h"

void matrix_print_k(dim3 grid_dim, dim3 block_dim, size_t argc, void** argv);

#endif
