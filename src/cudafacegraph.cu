#include "cudafacegraph.h"

#include <thrust/device_vector.h>

__global__ void __get_vertex_to_adj() {
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

    int triangles_per_block = 1024;
    int iter = (int)ceil((float)triangles->size() / triangles_per_block);

    // 로컬 adj 생성.
    int adj_max = 20;
    int* local_adj_map = (int*)malloc(iter * vertices_index * adj_max * sizeof(int));
    int* local_adj_map_index = (int*)malloc(iter * vertices_index * sizeof(int));

    memset(local_adj_map_index, 0, iter * vertices_index * sizeof(int));

    // 로컬 adj 연산.
    for (int i_iter = 0; i_iter < iter; i_iter++) {
        int* local_map = &local_adj_map[i_iter * (vertices_index + adj_max)];
        int* local_index = &local_adj_map_index[i_iter * vertices_index];

        int i_begin = i_iter * triangles_per_block;
        int i_end = (i_iter + 1) * triangles_per_block;

        for (int i = i_begin; i < triangles->size() && i < i_end; i++) {
            for (int j = 0; j < 3; j++) {
                int tri_id = triangles->at(i).id[j];
                local_map[tri_id * adj_max + local_index[tri_id]++] = i;
            }
        }
    }

    // 로컬 adj 취합.
    std::vector<std::vector<int>> vertex_adjacent_map(vertices_index);

    for (int i_iter = 0; i_iter < iter; i_iter++) {
        int* local_map = &local_adj_map[i_iter * (vertices_index + adj_max)];
        int* local_index = &local_adj_map_index[i_iter * vertices_index];

        for (int i = 0; i < vertices_index; i++) {
            vertex_adjacent_map[i].insert(vertex_adjacent_map[i].begin(), &local_map[i * adj_max], &local_map[i * adj_max + local_index[i]]);
        }
    }

    free(local_adj_map);
    free(local_adj_map_index);

    return vertex_adjacent_map;
}