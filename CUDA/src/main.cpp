#include <stdio.h>
#include <omp.h>

#include "header.h"
#include "cudaheader.h"

int main() {
    printf("Hello, world!\n");

    foo();

    #pragma omp parallel num_threads(4)
    {
        printf("Hello from OpenMP thread %d!\n", omp_get_thread_num());
    }

    int count = 10;
    float value = 3.14f;

    int* d_count;
    float* d_value;
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_value, sizeof(float));

    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice);

    void* args[] = { d_count, d_value };

    void** d_args;
    cudaMalloc(&d_args, 2 * sizeof(void*));

    cudaMemcpy(d_args, args, 2 * sizeof(void*), cudaMemcpyHostToDevice);

    kernel_call(foo_cuda, dim3(1, 1, 1), dim3(4, 1, 1), d_args);

    return 0;
}
