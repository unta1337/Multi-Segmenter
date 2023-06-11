#include "cudafacegraphutils.h"

// cuda 관련 헤더를 .h 등 .cu가 아닌 파일에서 include하면 에러 발생.
#include <cuda/semaphore>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void __segment_union_to_obj(glm::vec3** vertices, glm::ivec3** faces, int* group_id, Triangle* triangles,
                                       size_t triangles_count, size_t total_vertex_count, int* index_lookup_chunk,
                                       int* vertex_index_out, int* index_index_out) {
    __shared__ int vertex_index;    // push_back 대신 유지하는 정점 인덱스 추적 변수.
    __shared__ int index_index;     // push_back 대신 유지하는 삼각형 인덱스 추적 변수.
    __shared__ int* index_lookup;   // 기존 unordered_map을 유지하는 중복 검사용 변수.
    __shared__ cuda::binary_semaphore<cuda::thread_scope_block>* vertex_sem;     // 정점 삽입 mutex.
    __shared__ cuda::binary_semaphore<cuda::thread_scope_block>* index_sem;      // 삼각형 삽입 mutex.

    if (threadIdx.x == 0) {
        vertex_index = 0;
        index_index = 0;
        index_lookup = &index_lookup_chunk[blockIdx.x * total_vertex_count];
        vertex_sem = new cuda::binary_semaphore<cuda::thread_scope_block>();
        index_sem = new cuda::binary_semaphore<cuda::thread_scope_block>();
        vertex_sem->release();
        index_sem->release();
    }
    __syncthreads();

    for (int i = threadIdx.x; i < triangles_count; i += blockDim.x) {
        if (group_id[i] != blockIdx.x)
            continue;

        glm::ivec3 new_index;
        for (int j = 0; j < 3; j++) {
            int& index_if_exist = index_lookup[triangles[i].id[j]];

            vertex_sem->acquire();
            if (index_if_exist == -1) {
                vertices[blockIdx.x][vertex_index] = triangles[i].vertex[j];
                index_if_exist = ++vertex_index;
            }
            vertex_sem->release();

            new_index[j] = index_if_exist;
        }

        index_sem->acquire();
        faces[blockIdx.x][index_index] = new_index;
        index_index++;
        index_sem->release();
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        vertex_index_out[blockIdx.x] = vertex_index;
        index_index_out[blockIdx.x] = index_index;
        delete vertex_sem;
        delete index_sem;
    }
}

std::vector<TriangleMesh*> segment_union_to_obj(const std::vector<int> segment_union,
                                                const std::vector<Triangle>* triangles, size_t total_vertex_count) {
    std::vector<TriangleMesh*> result;
    std::vector<int> group_id(segment_union.size(), -1);    // 특정 요소가 속한 그룹 id.
    std::vector<int> group_count;                           // 특정 그룹의 요소 개수.

    int group_index = 0;
    for (int i = 0; i < segment_union.size(); i++) {
        int group_root = segment_union[i];
        int& g_id = group_id[group_root];

        if (g_id == -1) {
            result.push_back(new TriangleMesh);
            g_id = group_index++;
            result[g_id]->material = new Material;
            group_count.push_back(1);
        }

        group_id[i] = g_id;
        group_count[g_id]++;
    }

    // device 관련 변수 초기화.
    std::vector<thrust::device_vector<glm::vec3>> d_vertices(result.size(), thrust::device_vector<glm::vec3>());
    std::vector<thrust::device_vector<glm::ivec3>> d_faces(result.size(), thrust::device_vector<glm::ivec3>());

    thrust::device_vector<glm::vec3*> dd_vertices;
    thrust::device_vector<glm::ivec3*> dd_faces;

    for (int i = 0; i < result.size(); i++) {
        d_vertices[i].resize(triangles->size() * 3);
        d_faces[i].resize(triangles->size());

        dd_vertices.push_back(thrust::raw_pointer_cast(d_vertices[i].data()));
        dd_faces.push_back(thrust::raw_pointer_cast(d_faces[i].data()));
    }

    thrust::device_vector<int> d_group_id(group_id);
    thrust::device_vector<Triangle> d_triangles(*triangles);
    thrust::device_vector<int> d_index_lookup_chunk(group_index * total_vertex_count, -1);
    thrust::device_vector<int> d_vertex_index_out(group_index);
    thrust::device_vector<int> d_index_index_out(group_index);

    __segment_union_to_obj<<<group_count.size(), std::min(triangles->size(), (size_t)1024)>>>(thrust::raw_pointer_cast(dd_vertices.data()),
                                                                                              thrust::raw_pointer_cast(dd_faces.data()),
                                                                                              thrust::raw_pointer_cast(d_group_id.data()),
                                                                                              thrust::raw_pointer_cast(d_triangles.data()),
                                                                                              d_triangles.size(), total_vertex_count,
                                                                                              thrust::raw_pointer_cast(d_index_lookup_chunk.data()),
                                                                                              thrust::raw_pointer_cast(d_vertex_index_out.data()),
                                                                                              thrust::raw_pointer_cast(d_index_index_out.data()));
    cudaDeviceSynchronize();

    // Data Transfer: Device -> Host.
    thrust::host_vector<int> vertex_index_out(d_vertex_index_out);
    thrust::host_vector<int> index_index_out(d_index_index_out);

    for (int i = 0; i < result.size(); i++) {
        result[i]->vertex.resize(vertex_index_out[i]);
        result[i]->index.resize(index_index_out[i]);

        thrust::copy(d_vertices[i].begin(), d_vertices[i].begin() + vertex_index_out[i], result[i]->vertex.begin());
        thrust::copy(d_faces[i].begin(), d_faces[i].begin() + index_index_out[i], result[i]->index.begin());
    }
    cudaDeviceSynchronize();

    return result;
}
