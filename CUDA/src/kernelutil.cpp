#include "kernelutil.h"

void* to_dev_new(void* host_ptr, size_t size) {
    void* dev_ptr;
    cudaMalloc(&dev_ptr, size);
    cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice);

    return dev_ptr;
}
