#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "grouping.h"
#include "trianglemesh.hpp"
#include <dstimer.hpp>
#include <glm/gtx/normal.hpp>
#include <iostream>
#include <omp.h>
#include <stdlib.h>

/*
    mesh
std::vector<glm::vec3> vertex;
std::vector<glm::ivec3> index;
*/

void compare(TriangleMesh* mesh) { // 뭐가 더 좋을지 고민 용 함수.
    DS_timer timer(8);
    timer.setTimerName(0, (char*)"vertex malloc");
    timer.setTimerName(1, (char*)"index malloc");
    timer.setTimerName(2, (char*)"index transfer HtoD");
    timer.setTimerName(3, (char*)"index transfer HtoD");
    timer.setTimerName(4, (char*)"-------------------");
    timer.setTimerName(5, (char*)"openMP vertex align");
    timer.setTimerName(6, (char*)"vertex align malloc");
    timer.setTimerName(7, (char*)"vertex align transfer HtoD");

    // -------------------------------------------------------------------------------------

    glm::vec3* dVertexList;
    glm::ivec3* dIndexList;

    timer.onTimer(0);
    cudaMalloc(&dVertexList, mesh->vertex.size() * sizeof(glm::vec3));
    cudaDeviceSynchronize();
    timer.offTimer(0);
    timer.onTimer(1);
    cudaMalloc(&dIndexList, mesh->index.size() * sizeof(glm::ivec3));
    cudaDeviceSynchronize();
    timer.offTimer(1);

    timer.onTimer(2);
    cudaMemcpy(dVertexList, &mesh->vertex[0], mesh->vertex.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.offTimer(2);
    timer.onTimer(3);
    cudaMemcpy(dIndexList, &mesh->index[0], mesh->index.size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.offTimer(3);

    cudaDeviceSynchronize();
    // --------------------------------------------------------------------------------------------------------

    glm::vec3* dVertexAlign;

    timer.onTimer(5);
    Triangle* vectorList = (Triangle*)malloc(sizeof(Triangle) * mesh->index.size());
#pragma omp parallel for
    for (int i = 0; i < mesh->index.size(); i++) {
        vectorList[i].vertex[0] = mesh->vertex[mesh->index[i].x];
        vectorList[i].vertex[1] = mesh->vertex[mesh->index[i].y];
        vectorList[i].vertex[2] = mesh->vertex[mesh->index[i].z];
    }
    timer.offTimer(5);

    timer.onTimer(6);
    cudaMalloc(&dVertexAlign, sizeof(Triangle) * mesh->index.size());
    cudaDeviceSynchronize();
    timer.offTimer(6);

    timer.onTimer(7);
    cudaMemcpy(dVertexAlign, vectorList, sizeof(Triangle) * mesh->index.size(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.offTimer(7);

    // --------------------------------------------------------------------------------------------------------

    // kernel<<<1, 1, NULL, stream>>>();
    timer.printTimer();
    // cudaStreamDestroy(stream);
}

#define PI 3.14
__global__ void kernel(Triangle* dVertexAlign, unsigned int* group, size_t indexSize, float tolerance) {

    size_t threadId = threadIdx.x + (blockIdx.x * blockDim.x);
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

    xSeta += 15;
    ySeta += 15;
    zSeta += 15;
    unsigned int bitmap = (unsigned int)(xSeta / tolerance) % 360;
    bitmap = bitmap << 8;
    bitmap += (unsigned int)(ySeta / tolerance) % 360;
    bitmap = bitmap << 8;
    bitmap += (unsigned int)(zSeta / tolerance) % 360;
    group[threadId] = bitmap;
}

std::unordered_map<unsigned int, std::vector<Triangle>> kernelCall(TriangleMesh* mesh, float tolerance) {
    // compare(mesh);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    DS_timer timer(1);

    timer.onTimer(0);
    Triangle* dVertexAlign;
    unsigned int* dGroup;
    Triangle* vectorList = (Triangle*)malloc(sizeof(Triangle) * mesh->index.size());
    unsigned int* group = (unsigned int*)malloc(sizeof(unsigned int) * mesh->index.size());
#pragma omp parallel for
    for (int i = 0; i < mesh->index.size(); i++) {
        vectorList[i].vertex[0] = mesh->vertex[mesh->index[i].x];
        vectorList[i].vertex[1] = mesh->vertex[mesh->index[i].y];
        vectorList[i].vertex[2] = mesh->vertex[mesh->index[i].z];
    }
    cudaMallocAsync(&dGroup, sizeof(unsigned int) * mesh->index.size(), stream);
    cudaMalloc(&dVertexAlign, sizeof(Triangle) * mesh->index.size());
    cudaMemcpy(dVertexAlign, vectorList, sizeof(Triangle) * mesh->index.size(), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
#define BLOCK_SIZE 512
    kernel<<<ceil((float)mesh->index.size() / BLOCK_SIZE), BLOCK_SIZE>>>(dVertexAlign, dGroup, mesh->index.size(),
                                                                         tolerance);

    unsigned int size = (unsigned int)(360 / tolerance);
    std::unordered_map<unsigned int, std::vector<Triangle>> normal_triangle_list_map;
    std::unordered_map<unsigned int, omp_lock_t> lock_list;

    for (size_t i = 0; i <= size; i++) {
        unsigned int bitmap_i = (unsigned int)i;
        bitmap_i = bitmap_i << 16;
        for (size_t j = 0; j <= size; j++) {
            unsigned int bitmap_j = (unsigned int)j;
            bitmap_j = bitmap_j << 8;
            for (size_t k = 0; k <= size; k++) {
                unsigned int bitmap = (unsigned int)k;
                bitmap = bitmap_i | bitmap_j | bitmap;
                lock_list.insert({bitmap, omp_lock_t()});
                omp_init_lock(&lock_list[bitmap]);
            }
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(group, dGroup, sizeof(unsigned int) * mesh->index.size(), cudaMemcpyDeviceToHost);

#pragma omp parallel for
    for (int i = 0; i < mesh->index.size(); i++) {
        omp_set_lock(&lock_list[group[i]]);
        if (normal_triangle_list_map.find(group[i]) == normal_triangle_list_map.end())
            normal_triangle_list_map.insert({group[i], std::vector<Triangle>()});
        normal_triangle_list_map[group[i]].push_back(vectorList[i]);
        omp_unset_lock(&lock_list[group[i]]);
    }

    timer.offTimer(0);
    timer.printTimer();

    return normal_triangle_list_map;
}
