#include "cudafacegraph.h"

#define ADJ_MAX 20
#define BLOCK_LEN 512

__global__ void __get_vertex_to_adj(int* vertex_adj, Triangle* triangles, int triangle_count, int* vertex_count_out, int vertex_index_begin, int adj_max) {
    __shared__ int vertex_index;
    __shared__ int index;
    __shared__ int internal_index;
    __shared__ int s_vertex_adj[ADJ_MAX];
    __shared__ glm::vec3 cache[BLOCK_LEN];

    if (threadIdx.x == 0) {
        vertex_index = vertex_index_begin + blockIdx.x;
        index = 0;
        internal_index = blockIdx.x;
    }
    __syncthreads();

    for (int j = 0; j < 3; j++) {
        for (int i = threadIdx.x; i < triangle_count; i += blockDim.x) {
            if (triangles[i].id[j] != vertex_index)
                continue;

            s_vertex_adj[atomicAdd(&index, 1)] = i;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < index; i++)
            vertex_adj[internal_index * adj_max + i] = s_vertex_adj[i];
        vertex_count_out[internal_index] = index;
    }
}

std::vector<std::vector<int>> CUDAFaceGraph::get_vertex_to_adj() {
    d_triangles = thrust::device_vector<Triangle>(*triangles);
    std::vector<std::vector<int>> vertex_adjacent_map(total_vertex_count);

    int adj_max = ADJ_MAX;
    int batch_size = 8192;
    int iter = (int)ceil((float)total_vertex_count / batch_size);

    std::vector<cudaStream_t> streams(iter);

    for (cudaStream_t& stream : streams)
        cudaStreamCreate(&stream);

    std::vector<int*> vertex_adj(iter);
    std::vector<int*> vertex_count(iter);

    std::vector<int*> d_vertex_adj(iter);
    std::vector<int*> d_vertex_count(iter);

    // 동적 할당.
    for (int i = 0; i < iter; i++) {
        cudaMallocAsync(&d_vertex_adj[i], batch_size * adj_max * sizeof(int), streams[i]);
        cudaMallocAsync(&d_vertex_count[i], batch_size * sizeof(int), streams[i]);
    }

    // 동적 할당 host.
    #pragma omp parallel for
    for (int i = 0; i < iter; i++) {
        cudaMallocHost(&vertex_adj[i], batch_size * adj_max * sizeof(int));
        cudaMallocHost(&vertex_count[i], batch_size * sizeof(int));
    }

    cudaDeviceSynchronize();

    // 연산.
    for (int i = 0; i < iter; i++) {
        __get_vertex_to_adj<<<batch_size, std::min(triangles->size(), (size_t)BLOCK_LEN), 0, streams[i]>>>(d_vertex_adj[i],
                                                                                                           thrust::raw_pointer_cast(d_triangles.data()), d_triangles.size(),
                                                                                                           d_vertex_count[i], i * batch_size, adj_max);
    }
    cudaDeviceSynchronize();

    // 데이터 복사 1.
    for (int i = 0; i < iter; i++) {
        cudaMemcpyAsync(vertex_adj[i], d_vertex_adj[i], batch_size * adj_max * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(vertex_count[i], d_vertex_count[i], batch_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();

    // 동적 할당 해제.
    for (int i = 0; i < iter; i++) {
        cudaFreeAsync(&d_vertex_adj, streams[i]);
        cudaFreeAsync(&d_vertex_count, streams[i]);
    }

    // 데이터 복사 2.
    #pragma omp parallel for
    for (int i = 0; i < iter; i++) {
        for (int j = 0; j < batch_size; j++) {
            int index = i * batch_size + j;
            if (index < total_vertex_count)
                vertex_adjacent_map[index].insert(vertex_adjacent_map[index].begin(), vertex_adj[i] + (j * adj_max), vertex_adj[i] + (j * adj_max) + vertex_count[i][j]);
        }
    }

    // 동적 할당 해제 host.
    #pragma omp parallel for
    for (int i = 0; i < iter; i++) {
        cudaFreeHost(&vertex_adj);
        cudaFreeHost(&vertex_count);
    }

    cudaDeviceSynchronize();

    for (cudaStream_t& stream : streams)
        cudaStreamDestroy(stream);

    return vertex_adjacent_map;
}