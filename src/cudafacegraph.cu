#include "cudafacegraph.h"

#include <thrust/device_vector.h>

#define ADJ_MAX 20
#define BLOCK_LEN 1024

__global__ void __get_vertex_to_adj(int* local_adj_map, int* local_adj_map_index,
                                    Triangle* triangles, int triangle_count, int triangles_per_block,
                                    int vertex_count, int i_iter) {
    int* local_map = &local_adj_map[i_iter * (vertex_count + ADJ_MAX)];
    int* local_index = &local_adj_map_index[i_iter * vertex_count];
    int i_begin = i_iter * triangles_per_block;

    for (int i = threadIdx.x; i + i_begin < triangle_count && i < triangles_per_block; i += BLOCK_LEN) {
        for (int j = 0; j < 3; j++) {
            int tri_id = triangles[i].id[j];
            local_map[tri_id * ADJ_MAX + atomicAdd(&local_index[tri_id], 1)] = i + i_begin;
        }
    }
}

std::vector<std::vector<int>> CUDAFaceGraph::get_vertex_to_adj() {
    // Vertex Lookup vertices.
    vertices = std::vector<glm::vec3>(triangles->size() * 3);        // 정점 룩업 테이블.
    size_t vertices_index = 0;                                       // 테이블 인덱스.
    std::vector<int> new_index_lookup(total_vertex_count, -1);       // 인덱스 룩업 테이블.

    // 모든 정점을 순회하면서,
    for (int i = 0; i < triangles->size(); i++) {
        for (int j = 0; j < 3; j++) {
            size_t& tri_id = triangles->at(i).id[j];

            // 새로운 정점을 발견하면 테이블에 추가 및 id 갱신,
            if (new_index_lookup[tri_id] == -1) {
                tri_id = new_index_lookup[tri_id] = vertices_index;
                vertices[vertices_index] = triangles->at(i).vertex[j];
                vertices_index++;
            }

            // 이미 추가한 정점이면 id만 갱신.
            else {
                tri_id = new_index_lookup[tri_id];
            }
        }
    }
    vertices.resize(vertices_index);

    // 이제 각 FaceGraph에 속한 정점은 새로운 고유 번호를 갖게 됨.
    // 이로써 원래 obj에서의 인덱스를 고유 번호로 사용하지 않아도 됨.
    // 이후 정점 룩업 시 저장 공간 및 탐색 시간이 줄어듦.

    // TODO: triangles_per_block을 triangles->size()보다 작게 하면 오류 발생.
    int triangles_per_block = triangles->size();
    int iter = (int)ceil((float)triangles->size() / triangles_per_block);

    // 쿠다 스트림 생성.
    std::vector<cudaStream_t> streams(iter);
    for (cudaStream_t& stream : streams)
        cudaStreamCreate(&stream);

    // 로컬 adj 생성. (Host)
    int* local_adj_map; cudaMallocHost(&local_adj_map, iter * vertices_index * ADJ_MAX * sizeof(int));
    int* local_adj_map_index; cudaMallocHost(&local_adj_map_index, iter * vertices_index * sizeof(int));

    // 로컬 adj 생성. (Cuda)
    int* d_local_adj_map; cudaMalloc(&d_local_adj_map, iter * vertices_index * ADJ_MAX * sizeof(int));
    int* d_local_adj_map_index; cudaMalloc(&d_local_adj_map_index, iter * vertices_index * sizeof(int));

    // memset async.
    for (int i = 0; i < iter; i++)
        cudaMemsetAsync(&d_local_adj_map_index[i * vertices_index], 0, vertices_index * sizeof(int), streams[i]);
    thrust::device_vector<Triangle> d_triangles_vec(*triangles);
    Triangle* d_triangles = thrust::raw_pointer_cast(d_triangles_vec.data());

    cudaDeviceSynchronize();        // memset 동기화.

    // 연산 async.
    for (int i = 0; i < iter; i++) {
        __get_vertex_to_adj<<<1, BLOCK_LEN, 0, streams[i]>>>(d_local_adj_map, d_local_adj_map_index,
                                                             d_triangles, d_triangles_vec.size(), triangles_per_block,
                                                             vertices_index, i);
    }
    cudaDeviceSynchronize();        // 연산 동기화.

    // memcpy async.
    for (int i = 0; i < iter; i++) {
        cudaMemcpyAsync(&local_adj_map[i * (vertices_index + ADJ_MAX)], &d_local_adj_map[i * (vertices_index + ADJ_MAX)],
                        vertices_index * ADJ_MAX * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&local_adj_map_index[i * vertices_index], &d_local_adj_map_index[i * vertices_index],
                        vertices_index * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();        // memcpy 동기화.

    // free.
    cudaFree(d_local_adj_map);
    cudaFree(d_local_adj_map_index);

    // 로컬 adj 취합.
    std::vector<std::vector<int>> vertex_adjacent_map(vertices_index);

    for (int i_iter = 0; i_iter < iter; i_iter++) {
        int* local_map = &local_adj_map[i_iter * (vertices_index + ADJ_MAX)];
        int* local_index = &local_adj_map_index[i_iter * vertices_index];

        for (int i = 0; i < vertices_index; i++) {
            vertex_adjacent_map[i].insert(vertex_adjacent_map[i].begin(), &local_map[i * ADJ_MAX], &local_map[i * ADJ_MAX + local_index[i]]);
        }
    }

    // free.
    cudaFreeHost(local_adj_map);
    cudaFreeHost(local_adj_map_index);

    // 쿠다 스트림 종료.
    for (cudaStream_t& stream : streams)
        cudaStreamDestroy(stream);

    return vertex_adjacent_map;
}