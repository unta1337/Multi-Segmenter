#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void vectorAdd(const int *a, const int *b, int *c, int size) {
    unsigned int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < size) {
        c[tID] = a[tID] + b[tID];
    }
}