#include "cudafacegraphutils.h"

// cuda 관련 헤더를 .h 등 .cu가 아닌 파일에서 include하면 에러 발생.
#include <cuda/semaphore>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void __segment_union_to_obj(glm::vec3* vertices, glm::ivec3* faces, int* group_id, Triangle* triangles,
                                       size_t triangles_count, size_t total_vertex_count, int* index_lookup_chunk, int g_id,
                                       int* vertex_index_out, int* index_index_out) {
    __shared__ int vertex_index;    // push_back 대신 유지하는 정점 인덱스 추적 변수.
    __shared__ int index_index;     // push_back 대신 유지하는 삼각형 인덱스 추적 변수.
    __shared__ int* index_lookup;   // 기존 unordered_map을 유지하는 중복 검사용 변수.
    __shared__ cuda::binary_semaphore<cuda::thread_scope_block>* vertex_sem;     // 정점 삽입 mutex.

    if (threadIdx.x == 0) {
        vertex_index = 0;
        index_index = 0;
        index_lookup = index_lookup_chunk;
        vertex_sem = new cuda::binary_semaphore<cuda::thread_scope_block>();
        vertex_sem->release();
    }
    __syncthreads();

    for (int i = threadIdx.x; i < triangles_count; i += blockDim.x) {
        if (group_id[i] != g_id)
            continue;

        glm::ivec3 new_index;
        for (int j = 0; j < 3; j++) {
            int& index_if_exist = index_lookup[triangles[i].id[j]];

            vertex_sem->acquire();
            if (index_if_exist == -1) {
                vertices[vertex_index] = triangles[i].vertex[j];
                index_if_exist = ++vertex_index;
            }
            vertex_sem->release();

            new_index[j] = index_if_exist;
        }

        faces[atomicAdd(&index_index, 1) - 1] = new_index;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        *vertex_index_out = vertex_index;
        *index_index_out = index_index;
        delete vertex_sem;
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

    std::vector<cudaStream_t> streams(group_index);

    // cuda host.
    std::vector<glm::vec3*> vertex_out(group_index);
    std::vector<glm::ivec3*> face_out(group_index);
    int* vertex_index_out;
    int* face_index_out;

    for (int i = 0; i < group_index; i++) cudaMallocHost(&vertex_out[i], triangles->size() * 3 * sizeof(glm::vec3));
    for (int i = 0; i < group_index; i++) cudaMallocHost(&face_out[i], triangles->size() * sizeof(glm::ivec3));
    cudaMallocHost(&vertex_index_out, group_index * sizeof(int));
    cudaMallocHost(&face_index_out, group_index * sizeof(int));

    // 공통.
    thrust::device_vector<int> d_group_id(group_id);
    thrust::device_vector<Triangle> d_triangles(*triangles);

    for (cudaStream_t& stream : streams)
        cudaStreamCreate(&stream);

    #pragma omp parallel for
    for (int i = 0; i < group_index; i++) {
        cudaStream_t& stream = streams[i];

        glm::vec3* d_vertices;
        glm::ivec3* d_faces;

        int* d_index_lookup;
        int* d_vertex_index_out;
        int* d_face_index_out;

        cudaMallocAsync(&d_vertices, triangles->size() * 3 * sizeof(glm::vec3), stream);
        cudaMallocAsync(&d_faces, triangles->size() * sizeof(glm::ivec3), stream);

        cudaMallocAsync(&d_index_lookup, total_vertex_count * sizeof(int), stream);
        cudaMallocAsync(&d_vertex_index_out, sizeof(int), stream);
        cudaMallocAsync(&d_face_index_out, sizeof(int), stream);
        cudaStreamSynchronize(stream);      // 동적 할당 동기화.

        cudaMemsetAsync(d_index_lookup, 0xFF, total_vertex_count * sizeof(int), stream);
        cudaStreamSynchronize(stream);      // 메모리 초기화 동기화.

        __segment_union_to_obj<<<1, std::min(triangles->size(), (size_t)1024), 0, stream>>>(d_vertices,
                                                                                            d_faces,
                                                                                            thrust::raw_pointer_cast(d_group_id.data()),
                                                                                            thrust::raw_pointer_cast(d_triangles.data()),
                                                                                            d_triangles.size(), total_vertex_count,
                                                                                            d_index_lookup,
                                                                                            i,
                                                                                            d_vertex_index_out,
                                                                                            d_face_index_out);
        cudaStreamSynchronize(stream);      // 연산 동기화.

        cudaMemcpyAsync(&vertex_index_out[i], d_vertex_index_out, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&face_index_out[i], d_face_index_out, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);      // device -> host 동기화.

        cudaMemcpyAsync(vertex_out[i], d_vertices, vertex_index_out[i] * sizeof(glm::vec3), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(face_out[i], d_faces, face_index_out[i] * sizeof(glm::ivec3), cudaMemcpyDeviceToHost, stream);

        cudaFreeAsync(d_vertices, stream);
        cudaFreeAsync(d_faces, stream);

        cudaFreeAsync(d_index_lookup, stream);
        cudaFreeAsync(d_vertex_index_out, stream);
        cudaFreeAsync(d_face_index_out, stream);
    }
    cudaDeviceSynchronize();      // 최종 동기화.

    for (cudaStream_t& stream : streams)
        cudaStreamDestroy(stream);

    for (int i = 0; i < result.size(); i++) {
        result[i]->vertex.insert(result[i]->vertex.begin(), vertex_out[i], vertex_out[i] + vertex_index_out[i]);
        result[i]->index.insert(result[i]->index.begin(), face_out[i], face_out[i] + face_index_out[i]);
    }

    for (int i = 0; i < group_index; i++) cudaFreeHost(&vertex_out[i]);
    for (int i = 0; i < group_index; i++) cudaFreeHost(&face_out[i]);
    cudaFreeHost(vertex_index_out);
    cudaFreeHost(face_index_out);

    return result;
}
