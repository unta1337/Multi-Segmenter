#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "grouping.h"
#include "trianglemesh.hpp"
#include <glm/gtx/normal.hpp>
#include <omp.h>
#include <cstdlib>
#include <algorithm>

#define BLOCK_SIZE 512
#define TIMER_PREPROCESSING 0
#define TIMER_NORMAL_VECTOR_COMPUTATION 1
#define TIMER_MAP_COUNT 2
#define TIMER_NORMAL_MAP_INSERTION 3
#define TIMER_TOTAL 11

struct Pair {
    unsigned int first;  // group id
    unsigned int second; // TriangleList index

    __host__ __device__ bool operator()(const Pair& a, const Pair& b) const {
        if (a.first < b.first)
            return true;
        return false;
    }
};

#define PI 3.14
__global__ void grouping(Triangle* dVertexAlign, Pair* group, unsigned int indexSize, float tolerance) {

    unsigned int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (threadId >= indexSize)
        return;
    glm::vec3 normal = glm::normalize(glm::triangleNormal(
        dVertexAlign[threadId].vertex[0], dVertexAlign[threadId].vertex[1], dVertexAlign[threadId].vertex[2]));

    float xSeta = acosf(normal.z) / PI * 180;
    if (normal.z < 0.5f) // precision problem
        xSeta = 360 - xSeta;
    float ySeta = acosf(normal.x) / PI * 180;
    if (normal.x < 0.5f)
        ySeta = 360 - ySeta;
    float zSeta = acosf(normal.y) / PI * 180;
    if (normal.y < 0.5f)
        zSeta = 360 - zSeta;

    xSeta += 15; // 절대 각도 시작 위치 설정.
    ySeta += 15;
    zSeta += 15;

    unsigned int bitmap = (unsigned int)(xSeta / tolerance) % 360;
    bitmap = bitmap << 8;
    bitmap += (unsigned int)(ySeta / tolerance) % 360;
    bitmap = bitmap << 8;
    bitmap += (unsigned int)(zSeta / tolerance) % 360;

    group[threadId].first = bitmap;
    group[threadId].second = threadId;
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

std::unordered_map<unsigned int, std::vector<Triangle>> kernelCall(TriangleMesh* mesh, float tolerance,
                                                                   DS_timer& timer) {

    timer.onTimer(TIMER_PREPROCESSING);

    std::unordered_map<unsigned int, std::vector<Triangle>> normal_triangle_list_map;
    std::vector<Pair> hostData(mesh->index.size());
    Triangle* dVertexAlign;
    Pair* dGroup;
    Triangle* TriangleList = (Triangle*)malloc(sizeof(Triangle) * mesh->index.size());
    Pair* group = (Pair*)malloc(sizeof(Pair) * mesh->index.size());
    unsigned int* posList;
    unsigned int* dPos;
    unsigned int* dPosList;
    unsigned int pos = 0;

    // ------------------------------------- variable initial

    cudaMalloc(&dVertexAlign, sizeof(Triangle) * mesh->index.size());
    cudaMalloc(&dGroup, sizeof(Pair) * mesh->index.size());
    cudaMalloc(&dPos, sizeof(unsigned int));
    cudaMalloc(&dPosList, sizeof(unsigned int) * pow(360.f / tolerance, 3));

    cudaMemset(dPos, 0, sizeof(unsigned int));
    cudaMemset(dPosList, 0, sizeof(unsigned int) * pow(360.f / tolerance, 3));

    // ------------------------------------ Triangle Caculate

#pragma omp parallel for
    for (int i = 0; i < mesh->index.size(); i++) {
        TriangleList[i].vertex[0] = mesh->vertex[mesh->index[i].x];
        TriangleList[i].vertex[1] = mesh->vertex[mesh->index[i].y];
        TriangleList[i].vertex[2] = mesh->vertex[mesh->index[i].z];
    }

    cudaMemcpy(dVertexAlign, TriangleList, sizeof(Triangle) * mesh->index.size(), cudaMemcpyHostToDevice);

    // ----------------------------------- Vector Computation(grouping)

    timer.onTimer(TIMER_NORMAL_VECTOR_COMPUTATION);

    grouping<<<ceil((float)mesh->index.size() / BLOCK_SIZE), BLOCK_SIZE>>>(dVertexAlign, dGroup, mesh->index.size(),
                                                                           tolerance);
    cudaStreamSynchronize(0);

    timer.offTimer(TIMER_NORMAL_VECTOR_COMPUTATION);

    // ----------------------------------- Sort

    timer.onTimer(TIMER_MAP_COUNT);

    thrust::device_vector<Pair> deviceData(dGroup, dGroup + mesh->index.size());
    thrust::sort(deviceData.begin(), deviceData.end(), Pair());
    thrust::copy(deviceData.begin(), deviceData.end(), hostData.begin());

    splitIndex<<<ceil((float)mesh->index.size() / BLOCK_SIZE), BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(deviceData.data()), dPosList, dPos, mesh->index.size());

    cudaStreamSynchronize(0);

    posList = (unsigned int*)malloc(sizeof(unsigned int) * pow(360.f / tolerance, 3));

    cudaMemcpy(posList, dPosList, sizeof(unsigned int) * pow(360.f / tolerance, 3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pos, dPos, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    posList[pos] = 0;
    pos++;
    posList[pos] = mesh->index.size();
    pos++;

    std::sort(posList, posList + pos);

    timer.offTimer(TIMER_MAP_COUNT);

    // --------------------------------- Map Insertion

    timer.onTimer(TIMER_NORMAL_MAP_INSERTION);

    for (int i = 0; i < pos - 1; i++) {
        unsigned int start = posList[i];
        unsigned int end = posList[i + 1];
        unsigned int gid = hostData[start].first;
        normal_triangle_list_map.insert({gid, std::vector<Triangle>(end - start)});
    }

#pragma omp parallel for
    for (int i = 0; i < pos - 1; i++) {
        unsigned int start = posList[i];
        unsigned int end = posList[i + 1];
        unsigned int gid = hostData[start].first;

        for (unsigned int j = start; j < end; j++)
            normal_triangle_list_map[gid][j - start] = TriangleList[hostData[j].second];
    }

    timer.offTimer(TIMER_NORMAL_MAP_INSERTION);

    cudaFree(dVertexAlign);
    cudaFree(dGroup);
    cudaFree(dPos);
    cudaFree(dPosList);
    free(TriangleList);
    free(group);
    free(posList);

    timer.offTimer(TIMER_PREPROCESSING);

    return normal_triangle_list_map;
}
