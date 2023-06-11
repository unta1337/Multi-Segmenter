#include "cudafacegraphutils.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void __segment_union_to_obj(glm::vec3** vertices, glm::ivec3** faces, int* group_id, Triangle* triangles,
                                       size_t triangles_count, size_t total_vertex_count, int* index_lookup_chunk, int* vertex_index_out) {
    __shared__ int vertex_index;
    __shared__ int index_index;
    __shared__ int* index_lookup;

    if (threadIdx.x == 0) {
        vertex_index = 0;
        index_index = 0;
        index_lookup = &index_lookup_chunk[blockIdx.x * total_vertex_count];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < triangles_count; i += blockDim.x) {
        if (group_id[i] != blockIdx.x)
            continue;

        glm::ivec3 new_index;
        for (int j = 0; j < 3; j++) {
            int& index_if_exist = index_lookup[triangles[i].id[j]];

            if (index_if_exist == -1) {
                vertices[blockIdx.x][vertex_index] = triangles[i].vertex[j];
                atomicAdd(&vertex_index, 1);
                atomicExch(&index_if_exist, vertex_index);
            }

            new_index[j] = index_if_exist;
        }

        faces[blockIdx.x][index_index] = new_index;
        atomicAdd(&index_index, 1);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        vertex_index_out[blockIdx.x] = vertex_index;
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

    __segment_union_to_obj<<<group_count.size(), 1>>>(thrust::raw_pointer_cast(dd_vertices.data()),
                                                                      thrust::raw_pointer_cast(dd_faces.data()),
                                                                      thrust::raw_pointer_cast(d_group_id.data()),
                                                                      thrust::raw_pointer_cast(d_triangles.data()),
                                                                      d_triangles.size(), total_vertex_count,
                                                                      thrust::raw_pointer_cast(d_index_lookup_chunk.data()),
                                                                      thrust::raw_pointer_cast(d_vertex_index_out.data()));
    cudaDeviceSynchronize();

    thrust::host_vector<int> vertex_index_out(d_vertex_index_out);

    for (int i = 0; i < result.size(); i++) {
        result[i]->vertex.resize(vertex_index_out[i]);
        result[i]->index.resize(triangles->size());

        thrust::copy(d_vertices[i].begin(), d_vertices[i].begin() + vertex_index_out[i], result[i]->vertex.begin());
        thrust::copy(d_faces[i].begin(), d_faces[i].end(), result[i]->index.begin());
        cudaDeviceSynchronize();
    }

    return result;
}
