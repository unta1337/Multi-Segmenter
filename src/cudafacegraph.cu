#include "cudafacegraph.h"

#define VERT_ADJ_MAX 10
#define TRI_ADJ_MAX 10

__global__ void __get_vertex_to_adj(int* local_adj_map_chunk, int* local_adj_map_index_chunk,
                                    Triangle* triangles,
                                    int triangle_count, int triangle_per_block, int vertex_count) {
    int* local_map = &local_adj_map_chunk[blockIdx.x * vertex_count * VERT_ADJ_MAX];
    int* local_index = &local_adj_map_index_chunk[blockIdx.x * vertex_count];
    int i_begin = blockIdx.x * triangle_per_block;

    for (int i = threadIdx.x; i + i_begin < triangle_count && i < triangle_per_block; i += blockDim.x) {
        for (int j = 0; j < 3; j++) {
            size_t vert_id = triangles[i + i_begin].id[j];
            local_map[vert_id * VERT_ADJ_MAX + atomicAdd(&local_index[vert_id], 1)] = i + i_begin;
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

    cudaStream_t s_a, s_b, s_c;
    cudaStreamCreate(&s_a);
    cudaStreamCreate(&s_b);
    cudaStreamCreate(&s_c);

    int triangles_per_block = 8192;
    int iter = (int)ceil((float)triangles->size() / triangles_per_block);

    int* d_local_adj_map; cudaMallocAsync(&d_local_adj_map, iter * vertices_index * VERT_ADJ_MAX * sizeof(int), s_a);
    int* d_local_adj_index; cudaMallocAsync(&d_local_adj_index, iter * vertices_index * sizeof(int), s_b);
    Triangle* d_triangles; cudaMallocAsync(&d_triangles, triangles->size() * sizeof(Triangle), s_c);

    int* local_adj_map; cudaMallocHost(&local_adj_map,iter * vertices_index * VERT_ADJ_MAX * sizeof(int));
    int* local_adj_index; cudaMallocHost(&local_adj_index, iter * vertices_index * sizeof(int));

    cudaMemsetAsync(d_local_adj_index, 0, iter * vertices_index * sizeof(int), s_b);
    cudaMemcpyAsync(d_triangles, triangles->data(), triangles->size() * sizeof(Triangle), cudaMemcpyHostToDevice, s_c);

    cudaDeviceSynchronize();
    __get_vertex_to_adj<<<iter, 1024>>>(d_local_adj_map, d_local_adj_index,
                                        d_triangles,
                                        triangles->size(), triangles_per_block, vertices_index);
    cudaDeviceSynchronize();

    cudaMemcpyAsync(local_adj_map, d_local_adj_map, iter * vertices_index * VERT_ADJ_MAX * sizeof(int), cudaMemcpyDeviceToHost, s_a);
    cudaMemcpyAsync(local_adj_index, d_local_adj_index, iter * vertices_index * sizeof(int), cudaMemcpyDeviceToHost, s_b);

    std::vector<std::vector<int>> vertex_adjacent_map(vertices_index);

    cudaDeviceSynchronize();
    for (int i = 0; i < iter; i++) {
        int* local_map = &local_adj_map[i * vertices_index * VERT_ADJ_MAX];
        int* local_index = &local_adj_index[i * vertices_index];

        for (int j = 0; j < vertices_index; j++) {
            vertex_adjacent_map[j].insert(vertex_adjacent_map[j].end(),
                                          &local_map[j * VERT_ADJ_MAX],
                                          &local_map[j * VERT_ADJ_MAX + local_index[j]]);
        }
    }

    cudaFreeAsync(d_local_adj_map, s_a);
    cudaFreeAsync(d_local_adj_index, s_b);
    cudaFreeAsync(d_triangles, s_c);

    cudaFreeHost(local_adj_map);
    cudaFreeHost(local_adj_index);

    cudaStreamDestroy(s_a);
    cudaStreamDestroy(s_b);
    cudaStreamDestroy(s_c);

    return vertex_adjacent_map;
}

__device__ int __is_connected(const Triangle& a, const Triangle& b) {
    int shared_vertices = 0;

    for (auto i : a.id) {
        for (auto j : b.id) {
            if (i == j) {
                shared_vertices++;
                break;
            }
        }
    }

    return (shared_vertices > 1);
}

__global__ void __get_adj_triangles(int* local_adj_map, int* local_adj_map_index,
                                    Triangle* triangles, int triangle_count, int triangles_per_block,
                                    int* vertex_adjacent_map, int* vertex_adjacent_map_count) {
    int* local_map = &local_adj_map[blockIdx.x * triangle_count * TRI_ADJ_MAX];
    int* local_index = &local_adj_map_index[blockIdx.x * triangle_count];
    int i_begin = blockIdx.x * triangles_per_block;

    for (int i = threadIdx.x; i + i_begin < triangle_count && i < triangles_per_block; i += blockDim.x) {
        Triangle tri_i = triangles[i + i_begin];

        for (int j = 0; j < 3; j++) {
            int vert_id = tri_i.id[j];

            int* adjacent_triangles = &vertex_adjacent_map[vert_id * VERT_ADJ_MAX];
            int adjacent_triangle_count = vertex_adjacent_map_count[vert_id];

            for (int k = 0; k < adjacent_triangle_count; k++) {
                int adjacent_triangle = adjacent_triangles[k];
                Triangle tri_adj = triangles[adjacent_triangle];

                if (i + i_begin != adjacent_triangle && __is_connected(tri_i, tri_adj)) {
                    local_adj_map[(i + i_begin) * TRI_ADJ_MAX + atomicAdd(&local_index[i + i_begin], 1)] = adjacent_triangle;
                }
            }
        }
    }
}

std::vector<std::vector<int>> CUDAFaceGraph::get_adj_triangles(std::vector<std::vector<int>>& vertex_adjacent_map) {
    // TODO: triangles->size()보다 작게 했을 때 버그 발생.
    int triangles_per_block = triangles->size();
    int iter = (int)ceil((float)triangles->size() / triangles_per_block);

    int* local_adj_map; cudaMallocHost(&local_adj_map, iter * triangles->size() * TRI_ADJ_MAX * sizeof(int));
    int* local_adj_map_index; cudaMallocHost(&local_adj_map_index, iter * triangles->size() * sizeof(int));

    int* d_local_adj_map; cudaMalloc(&d_local_adj_map, iter * triangles->size() * TRI_ADJ_MAX * sizeof(int));
    int* d_local_adj_map_index; cudaMalloc(&d_local_adj_map_index, iter * triangles->size() * sizeof(int));
    cudaMemset(d_local_adj_map_index, 0, iter * triangles->size() * sizeof(int));

    Triangle* d_triangles; cudaMalloc(&d_triangles, triangles->size() * sizeof(Triangle));
    cudaMemcpy(d_triangles, triangles->data(), triangles->size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    int* d_vertex_adjacent_map; cudaMalloc(&d_vertex_adjacent_map, vertices.size() * VERT_ADJ_MAX * sizeof(int));
    int* d_vertex_adjacent_map_count; cudaMalloc(&d_vertex_adjacent_map_count, vertices.size() * sizeof(int));

    for (int i = 0; i < vertices.size(); i++) {
        int size = vertex_adjacent_map[i].size();
        cudaMemcpy(&d_vertex_adjacent_map[i * VERT_ADJ_MAX], (int*)vertex_adjacent_map[i].data(), size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_vertex_adjacent_map_count[i], &size, sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();
    __get_adj_triangles<<<iter, 1024>>>(d_local_adj_map, d_local_adj_map_index,
                                        d_triangles, triangles->size(), triangles_per_block,
                                        d_vertex_adjacent_map, d_vertex_adjacent_map_count);
    cudaDeviceSynchronize();

    cudaMemcpy(local_adj_map, d_local_adj_map, iter * triangles->size() * TRI_ADJ_MAX * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(local_adj_map_index, d_local_adj_map_index, iter * triangles->size() * sizeof(int), cudaMemcpyDeviceToHost);

    adj_triangles = std::vector<std::vector<int>>(triangles->size());

    cudaDeviceSynchronize();
    for (int i_iter = 0; i_iter < iter; i_iter++) {
        int* local_map = &local_adj_map[i_iter * triangles->size() * TRI_ADJ_MAX];
        int* local_index = &local_adj_map_index[i_iter * triangles->size()];

        for (int i = 0; i < triangles->size(); i++) {
            adj_triangles[i].insert(adj_triangles[i].end(),
                                    &local_map[i * TRI_ADJ_MAX],
                                    &local_map[i * TRI_ADJ_MAX + local_index[i]]);
        }
    }

    cudaFree(d_local_adj_map);
    cudaFree(d_local_adj_map_index);
    cudaFree(d_triangles);
    cudaFree(d_vertex_adjacent_map);
    cudaFree(d_vertex_adjacent_map_count);

    cudaFreeHost(local_adj_map);
    cudaFreeHost(local_adj_map_index);

    return adj_triangles;
}