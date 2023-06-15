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
#define PI 3.14
#define SPLIT_SIZE 5

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

__global__ void grouping(Triangle* dVertexAlign, Pair* group, unsigned int indexSize, float tolerance,
                         unsigned int startPos) {

    unsigned int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int saveIndex = startPos + threadIdx.x + (blockIdx.x * blockDim.x);
    if (saveIndex >= indexSize)
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

    group[saveIndex].first = bitmap;
    group[saveIndex].second = saveIndex;
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


    cudaStream_t streamForAlloc;
    cudaStreamCreate(&streamForAlloc);
    cudaStream_t streamForCopy;
    cudaStreamCreate(&streamForCopy);

    cudaEvent_t eventListForAlloc[SPLIT_SIZE];
    cudaEvent_t eventListForCopy[SPLIT_SIZE];
    for (int i = 0; i < SPLIT_SIZE; i++) {
        cudaEventCreate(&eventListForAlloc[i]);
        cudaEventCreate(&eventListForCopy[i]);
    }

    size_t indexSize = mesh->index.size();
    size_t calcSize = ceil((double)indexSize / SPLIT_SIZE);

    std::unordered_map<unsigned int, std::vector<Triangle>> normal_triangle_list_map;
    std::vector<Pair> hostData(indexSize);
    Triangle* dVertexAlign[SPLIT_SIZE];
    Pair* dGroup;
    Triangle* TriangleList = (Triangle*)malloc(sizeof(Triangle) * indexSize);
    Pair* group = (Pair*)malloc(sizeof(Pair) * indexSize);
    unsigned int* posList;
    unsigned int* dPos;
    unsigned int* dPosList;
    unsigned int pos = 0;

    // ------------------------------------- variable initial

    cudaMallocAsync(&dGroup, sizeof(Pair) * indexSize, streamForAlloc);
    for (int i = 0; i < SPLIT_SIZE; i++) {
        cudaMallocAsync(&dVertexAlign[i], sizeof(Triangle) * calcSize, streamForAlloc);
        cudaEventRecord(eventListForAlloc[i], streamForAlloc);
    }
    cudaMallocAsync(&dPos, sizeof(unsigned int), streamForAlloc);
    cudaMallocAsync(&dPosList, sizeof(unsigned int) * pow(360.f / tolerance, 3), streamForAlloc);
    cudaMemsetAsync(dPos, 0, sizeof(unsigned int), streamForAlloc);
    cudaMemsetAsync(dPosList, 0, sizeof(unsigned int) * pow(360.f / tolerance, 3), streamForAlloc);

    // ------------------------------------ Triangle Caculate

    timer.onTimer(TIMER_NORMAL_VECTOR_COMPUTATION);

#pragma omp parallel for
    for (int i = 0; i < indexSize; i++) {
        TriangleList[i].vertex[0] = mesh->vertex[mesh->index[i].x];
        TriangleList[i].vertex[1] = mesh->vertex[mesh->index[i].y];
        TriangleList[i].vertex[2] = mesh->vertex[mesh->index[i].z];
    }

    // ----------------------------------- Vector Computation(grouping)

    size_t memCpyStart = 0;
    size_t groupingStart = 0;
    while (true) {

        for (size_t i = memCpyStart; i < SPLIT_SIZE; i++) {
            if (cudaEventQuery(eventListForAlloc[i]) == cudaSuccess) {
                if (i != SPLIT_SIZE - 1)
                    cudaMemcpyAsync(dVertexAlign[i], &TriangleList[calcSize * i], sizeof(Triangle) * calcSize,
                                    cudaMemcpyHostToDevice, streamForCopy);
                else
                    cudaMemcpyAsync(dVertexAlign[i], &TriangleList[calcSize * i],
                                    sizeof(Triangle) * mesh->index.size() % SPLIT_SIZE, cudaMemcpyHostToDevice,
                                    streamForCopy);

                cudaEventRecord(eventListForCopy[i], streamForCopy);
                memCpyStart = i;
                if (memCpyStart == SPLIT_SIZE - 1)
                    memCpyStart++;
            } else
                break;
        }

        for (size_t i = groupingStart; i < SPLIT_SIZE; i++) {
            if (cudaEventQuery(eventListForCopy[i]) == cudaSuccess) {
                grouping<<<ceil((float)calcSize / BLOCK_SIZE), BLOCK_SIZE>>>(dVertexAlign[i], dGroup, indexSize,
                                                                             tolerance, calcSize * i);
                groupingStart = i;
                if (groupingStart == SPLIT_SIZE - 1)
                    groupingStart++;
            } else
                break;
        }

        if (memCpyStart == SPLIT_SIZE && groupingStart == SPLIT_SIZE)
            break;
    }

    cudaStreamSynchronize(0);

    timer.offTimer(TIMER_NORMAL_VECTOR_COMPUTATION);

    // ----------------------------------- Sort

    timer.onTimer(TIMER_MAP_COUNT);

    cudaStreamSynchronize(streamForAlloc);
    thrust::device_vector<Pair> deviceData(dGroup, dGroup + indexSize);
    thrust::sort(deviceData.begin(), deviceData.end(), Pair());
    thrust::copy(deviceData.begin(), deviceData.end(), hostData.begin());

    splitIndex<<<ceil((float)indexSize / BLOCK_SIZE), BLOCK_SIZE>>>(thrust::raw_pointer_cast(deviceData.data()),
                                                                    dPosList, dPos, indexSize);

    cudaStreamSynchronize(0);

    posList = (unsigned int*)malloc(sizeof(unsigned int) * pow(360.f / tolerance, 3));

    cudaMemcpy(posList, dPosList, sizeof(unsigned int) * pow(360.f / tolerance, 3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pos, dPos, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    posList[pos] = 0;
    pos++;
    posList[pos] = indexSize;
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

    
    for (int i = 0; i < SPLIT_SIZE; i++) {
        cudaFree(dVertexAlign[i]);
        cudaEventDestroy(eventListForAlloc[i]);
        cudaEventDestroy(eventListForCopy[i]);
    }
    cudaStreamDestroy(streamForAlloc);
    cudaStreamDestroy(streamForCopy);
    cudaFree(dGroup);
    cudaFree(dPos);
    cudaFree(dPosList);
    free(TriangleList);
    free(group);
    free(posList);

    timer.offTimer(TIMER_PREPROCESSING);

    return normal_triangle_list_map;
}
