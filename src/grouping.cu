#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "grouping.h"
#include "trianglemesh.hpp"
#include <dstimer.hpp>
#include <glm/gtx/normal.hpp>
#include <omp.h>
#include <stdlib.h>

#include <algorithm>

struct Pair {
    unsigned int first;  // group id
    unsigned int second; // TriangleList index

    __device__ Pair& operator=(const Pair& other) {
        if (this != &other) { // protect against invalid self-assignment
            first = other.first;
            second = other.second;
        }
        // by convention, always return *this
        return *this;
    }

    __device__ bool operator()(const Pair& a, const Pair& b) const {
        if (a.first < b.first)
            return true;
        return false;
    }
};

#define PI 3.14
__global__ void grouping(Triangle* dVertexAlign, Pair* group, unsigned int indexSize, float tolerance) {

}

__global__ void splitIndex(Pair* group, unsigned int* posList, unsigned int* size, unsigned int indexSize) {

    unsigned int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (threadId >= indexSize || threadId == 0)
        return;

    if (group[threadId].first != group[threadId - 1].first) {
        unsigned int prev = atomicAdd(size, 1);
        posList[prev] = threadId;
    }
}

std::unordered_map<unsigned int, std::vector<Triangle>> kernelCall(TriangleMesh* mesh, float tolerance) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::unordered_map<unsigned int, std::vector<Triangle>> normal_triangle_list_map;

    return normal_triangle_list_map;
}
