#ifndef __CUDAHEADER_H
#define __CUDAHEADER_H

#include <stdio.h>
#include <cuda_runtime.h>

extern void (*foo_cuda)(void**);

void kernel_call(void (*func)(void**), dim3 grid_dim, dim3 block_dim, void** args);

#endif
