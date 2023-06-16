#include "cudafacegraph.h"

#define VERT_ADJ_MAX 20
#define TRI_ADJ_MAX 20

__global__ void __get_vertex_to_adj(int* local_adj_map, int* local_adj_map_index,
                                    Triangle* triangles, int triangle_count, int triangles_per_block,
                                    int vertex_count, int i_iter) {
    int* local_map = &local_adj_map[i_iter * (vertex_count + VERT_ADJ_MAX)];
    int* local_index = &local_adj_map_index[i_iter * vertex_count];
    int i_begin = i_iter * triangles_per_block;

    for (int i = threadIdx.x; i + i_begin < triangle_count && i < triangles_per_block; i += blockDim.x) {
        for (int j = 0; j < 3; j++) {
            int tri_id = triangles[i].id[j];
            local_map[tri_id * VERT_ADJ_MAX + atomicAdd(&local_index[tri_id], 1)] = i + i_begin;
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

    // TODO: triangles_per_block을 triangles->size()보다 작게 하면 노이즈 발생.
    int triangles_per_block = triangles->size();
    int iter = (int)ceil((float)triangles->size() / triangles_per_block);

    // 쿠다 스트림 생성.
    std::vector<cudaStream_t> streams(iter);
    for (cudaStream_t& stream : streams)
        cudaStreamCreate(&stream);

    // 로컬 adj 생성. (Host)
    int* local_adj_map; cudaMallocHost(&local_adj_map, iter * vertices_index * VERT_ADJ_MAX * sizeof(int));
    int* local_adj_map_index; cudaMallocHost(&local_adj_map_index, iter * vertices_index * sizeof(int));

    // 로컬 adj 생성. (Cuda)
    int* d_local_adj_map; cudaMalloc(&d_local_adj_map, iter * vertices_index * VERT_ADJ_MAX * sizeof(int));
    int* d_local_adj_map_index; cudaMalloc(&d_local_adj_map_index, iter * vertices_index * sizeof(int));

    // memset async.
    for (int i = 0; i < iter; i++)
        cudaMemsetAsync(&d_local_adj_map_index[i * vertices_index], 0, vertices_index * sizeof(int), streams[i]);

    Triangle* d_triangles;
    cudaMalloc(&d_triangles, triangles->size() * sizeof(Triangle));
    cudaMemcpy(d_triangles, triangles->data(), triangles->size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();        // memset 동기화.

    // 연산 async.
    for (int i = 0; i < iter; i++) {
        __get_vertex_to_adj<<<1, 1024, 0, streams[i]>>>(d_local_adj_map, d_local_adj_map_index,
                                                             d_triangles, triangles->size(), triangles_per_block,
                                                             vertices_index, i);
    }
    cudaDeviceSynchronize();        // 연산 동기화.

    // memcpy async.
    for (int i = 0; i < iter; i++) {
        cudaMemcpyAsync(&local_adj_map[i * (vertices_index + VERT_ADJ_MAX)], &d_local_adj_map[i * (vertices_index + VERT_ADJ_MAX)],
                        vertices_index * VERT_ADJ_MAX * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&local_adj_map_index[i * vertices_index], &d_local_adj_map_index[i * vertices_index],
                        vertices_index * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();        // memcpy 동기화.

    // free.
    cudaFree(d_local_adj_map);
    cudaFree(d_local_adj_map_index);
    cudaFree(d_triangles);

    // 로컬 adj 취합.
    std::vector<std::vector<int>> vertex_adjacent_map(vertices_index);

    for (int i_iter = 0; i_iter < iter; i_iter++) {
        int* local_map = &local_adj_map[i_iter * (vertices_index + VERT_ADJ_MAX)];
        int* local_index = &local_adj_map_index[i_iter * vertices_index];

        for (int i = 0; i < vertices_index; i++) {
            vertex_adjacent_map[i].insert(vertex_adjacent_map[i].begin(), &local_map[i * VERT_ADJ_MAX], &local_map[i * VERT_ADJ_MAX + local_index[i]]);
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
                                    int* vertex_adjacent_map, int* vertex_adjacent_map_count, int i_iter, int vertex_count) {
    int* local_map = &local_adj_map[i_iter * (triangle_count + TRI_ADJ_MAX)];
    int* local_index = &local_adj_map_index[i_iter * triangle_count];
    int i_begin = i_iter * triangles_per_block;

    for (int i = threadIdx.x; i + i_begin < triangle_count && i < triangles_per_block; i += blockDim.x) {
        Triangle tri_i = triangles[i];

        for (int j = 0; j < 3; j++) {
            int tri_id = triangles[i].id[j];

            int* adjacent_triangles = &vertex_adjacent_map[tri_id * VERT_ADJ_MAX];
            int adjacent_triangle_count = vertex_adjacent_map_count[tri_id];

            for (int k = 0; k < adjacent_triangle_count; k++) {
                int adjacent_triangle = adjacent_triangles[k];
                Triangle tri_adj = triangles[adjacent_triangle];

                if (i != adjacent_triangle && __is_connected(tri_i, tri_adj)) {
                    local_adj_map[i * TRI_ADJ_MAX + atomicAdd(&local_index[i], 1)] = adjacent_triangle;
                }
            }
        }
    }
}

std::vector<std::vector<int>> CUDAFaceGraph::get_adj_triangles(std::vector<std::vector<int>>& vertex_adjacent_map) {
    // TODO: triangles_per_block을 triangles->size()보다 작게 하면 노이즈 발생.
    int triangles_per_block = triangles->size();
    int iter = (int)ceil((float)triangles->size() / triangles_per_block);

    std::vector<cudaStream_t> streams(iter);
    for (cudaStream_t& stream : streams)
        cudaStreamCreate(&stream);

    // 로컬 adj 생성. (Host)
    int* local_adj_map; cudaMallocHost(&local_adj_map, iter * triangles->size() * TRI_ADJ_MAX * sizeof(int));
    int* local_adj_map_index; cudaMallocHost(&local_adj_map_index, iter * triangles->size() * sizeof(int));

    // 로컬 adj 생성. (Cuda)
    int* d_local_adj_map; cudaMalloc(&d_local_adj_map, iter * triangles->size() * TRI_ADJ_MAX * sizeof(int));
    int* d_local_adj_map_index; cudaMalloc(&d_local_adj_map_index, iter * triangles->size() * sizeof(int));

    // memset async.
    for (int i = 0; i < iter; i++)
        cudaMemsetAsync(&d_local_adj_map_index[i * triangles->size()], 0, triangles->size() * sizeof(int), streams[i]);

    Triangle* d_triangles;
    cudaMalloc(&d_triangles, triangles->size() * sizeof(Triangle));
    cudaMemcpy(d_triangles, triangles->data(), triangles->size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    int* d_vertex_adjacent_map; cudaMalloc(&d_vertex_adjacent_map, vertices.size() * VERT_ADJ_MAX * sizeof(int));
    int* d_vertex_adjacent_map_count; cudaMalloc(&d_vertex_adjacent_map_count, vertices.size() * sizeof(int));

    for (int i = 0; i < vertices.size(); i++) {
        int size = vertex_adjacent_map[i].size();
        cudaMemcpy(&d_vertex_adjacent_map[i * VERT_ADJ_MAX], (int*)vertex_adjacent_map[i].data(), size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_vertex_adjacent_map_count[i], &size, sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();        // memset 동기화.

    // 연산 async.
    for (int i = 0; i < iter; i++) {
        __get_adj_triangles<<<1, 1024, 0, streams[i]>>>(d_local_adj_map, d_local_adj_map_index,
                                                        d_triangles, triangles->size(), triangles_per_block,
                                                        d_vertex_adjacent_map, d_vertex_adjacent_map_count, i, vertices.size());
    }
    cudaDeviceSynchronize();        // 연산 동기화.

    // memcpy async.
    for (int i = 0; i < iter; i++) {
        cudaMemcpyAsync(&local_adj_map[i * (triangles->size() + TRI_ADJ_MAX)], &d_local_adj_map[i * (triangles->size() + TRI_ADJ_MAX)],
                        triangles->size() * TRI_ADJ_MAX * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(&local_adj_map_index[i * triangles->size()], &d_local_adj_map_index[i * triangles->size()],
                        triangles->size() * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();        // memcpy 동기화.

    // free.
    cudaFree(d_local_adj_map);
    cudaFree(d_local_adj_map_index);
    cudaFree(d_triangles);

    // 로컬 adj 취합.
    adj_triangles = std::vector<std::vector<int>>(triangles->size());

    for (int i_iter = 0; i_iter < iter; i_iter++) {
        int* local_map = &local_adj_map[i_iter * (triangles->size() + TRI_ADJ_MAX)];
        int* local_index = &local_adj_map_index[i_iter * triangles->size()];

        for (int i = 0; i < triangles->size(); i++) {
            adj_triangles[i].insert(adj_triangles[i].begin(), &local_map[i * TRI_ADJ_MAX], &local_map[i * TRI_ADJ_MAX + local_index[i]]);
        }
    }

    // free.
    cudaFreeHost(local_adj_map);
    cudaFreeHost(local_adj_map_index);

    cudaFree(&d_vertex_adjacent_map);
    cudaFree(&d_vertex_adjacent_map_count);

    // 쿠다 스트림 종료.
    for (cudaStream_t& stream : streams)
        cudaStreamDestroy(stream);

    return adj_triangles;
}