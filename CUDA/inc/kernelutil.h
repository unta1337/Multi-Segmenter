#ifndef __KERNELUTIL_H
#define __KERNELUTIL_H

#include <cuda_runtime.h>

#define KERNEL_DEF(__type, __name, __body) \
__type void __name(size_t argc, void** argv) { \
    __body \
} \
void __name ## _k(dim3 __grid_dim, dim3 __block_dim, size_t __ARGC, void** __ARGV) { \
    void** __d_argv; \
    cudaMalloc(&__d_argv, __ARGC * sizeof(void*)); \
    cudaMemcpy(__d_argv, __ARGV, __ARGC * sizeof(void*), cudaMemcpyHostToDevice); \
    __name<<<__grid_dim, __block_dim>>>(__ARGC, __d_argv); \
    cudaFree(__d_argv); \
}

void* to_dev_new(void* host_ptr, size_t size);

#endif
