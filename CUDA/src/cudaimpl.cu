#include "cudaheader.h"

__global__ void __foo_cuda(void** args) {
    int count = *(int*)args[0];
    float value = *(float*)args[1];

    #pragma omp parallel num_threads(4)
    {
        printf("OpenMP in CUDA! This should be printed 4 time.\n");
    }

    printf("Hello from CUDA Kernel (%d %d %d), (%d %d %d) with args: %d, %f!\n",
            blockIdx.x, blockIdx.y, blockIdx.z,
            threadIdx.x, threadIdx.y, threadIdx.z,
            count, value);
}
void (*foo_cuda)(void**) = __foo_cuda;

void kernel_call(void (*func)(void**), dim3 grid_dim, dim3 block_dim, void** args) {
    func<<<grid_dim, block_dim>>>(args);
}
