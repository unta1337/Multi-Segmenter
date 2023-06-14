#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "grouping.h"
#include "trianglemesh.hpp"
#include <algorithm>
#include <cstdlib>
#include <glm/gtx/normal.hpp>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define TIMER_PREPROCESSING 0
#define TIMER_NORMAL_VECTOR_COMPUTATION 1
#define TIMER_MAP_COUNT 2
#define TIMER_NORMAL_MAP_INSERTION 3
#define TIMER_CC_N_TMG 4
#define TIMER_FACEGRAPH_INIT_A 5
#define TIMER_FACEGRAPH_INIT_B 6
#define TIMER_FACEGRAPH_GET_SETMENTS_A 7
#define TIMER_FACEGRAPH_GET_SETMENTS_B 8
#define TIMER_TRIANGLE_MESH_GENERATING 9
#define TIMER_SEGMENT_COLORING 10
#define TIMER_TOTAL 11
#define TIMER_SIZE (TIMER_TOTAL + 1)

#define cudaEventSynchronize(i)                                                                                        \
    while (cudaEventQuery(eventList[i]) != cudaSuccess)                                                                \
        ;

struct Pair {
    unsigned int first;  // group id
    unsigned int second; // TriangleList index

    __device__ bool operator()(const Pair& a, const Pair& b) const {
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t eventList[3];
    for (int i = 0; i < 3; i++)
        cudaEventCreate(&eventList[i]);

    std::unordered_map<unsigned int, std::vector<Triangle>> normal_triangle_list_map;
    Triangle* dVertexAlign;
    Pair* dGroup;
    Triangle* TriangleList = (Triangle*)malloc(sizeof(Triangle) * mesh->index.size());
    Pair* group = (Pair*)malloc(sizeof(Pair) * mesh->index.size());
    unsigned int* posList;
    unsigned int* dPos;
    unsigned int* dPosList;
    unsigned int pos = 0;

    cudaMallocAsync(&dVertexAlign, sizeof(Triangle) * mesh->index.size(), stream);
    cudaEventRecord(eventList[0], stream);
    cudaMallocAsync(&dGroup, sizeof(Pair) * mesh->index.size(), stream);
    cudaEventRecord(eventList[1], stream);
    cudaMallocAsync(&dPos, sizeof(unsigned int), stream);
    cudaMemsetAsync(dPos, 0, sizeof(unsigned int), stream);
    cudaMallocAsync(&dPosList, sizeof(unsigned int) * pow(360.f / tolerance, 3), stream);
    cudaMemsetAsync(dPosList, 0, sizeof(unsigned int) * pow(360.f / tolerance, 3), stream);
    cudaEventRecord(eventList[2], stream);

#pragma omp parallel for
    for (int i = 0; i < mesh->index.size(); i++) {
        TriangleList[i].vertex[0] = mesh->vertex[mesh->index[i].x];
        TriangleList[i].vertex[1] = mesh->vertex[mesh->index[i].y];
        TriangleList[i].vertex[2] = mesh->vertex[mesh->index[i].z];
    }
    cudaEventSynchronize(0);
    cudaMemcpy(dVertexAlign, TriangleList, sizeof(Triangle) * mesh->index.size(), cudaMemcpyHostToDevice);
    cudaEventSynchronize(1);
#define BLOCK_SIZE 512
    grouping<<<ceil((float)mesh->index.size() / BLOCK_SIZE), BLOCK_SIZE>>>(dVertexAlign, dGroup, mesh->index.size(),
                                                                           tolerance);

    std::vector<Pair> hostData(mesh->index.size());
    thrust::device_vector<Pair> deviceData(dGroup, dGroup + mesh->index.size());
    thrust::sort(deviceData.begin(), deviceData.end(), Pair());
    thrust::copy(deviceData.begin(), deviceData.end(), hostData.begin());

    cudaEventSynchronize(2);
    splitIndex<<<ceil((float)mesh->index.size() / BLOCK_SIZE), BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(deviceData.data()), dPosList, dPos, mesh->index.size());

    posList = (unsigned int*)malloc(sizeof(unsigned int) * pow(360.f / tolerance, 3));

    cudaMemcpy(posList, dPosList, sizeof(unsigned int) * pow(360.f / tolerance, 3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pos, dPos, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    posList[pos] = 0;
    pos++;
    posList[pos] = mesh->index.size();
    pos++;

    std::sort(posList, posList + pos);

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

    timer.offTimer(TIMER_PREPROCESSING);

    return normal_triangle_list_map;
}
