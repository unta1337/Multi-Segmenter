#include "cudafacegraphutils.h"

std::vector<TriangleMesh*> segment_union_to_obj(const std::vector<int> segment_union,
                                                const std::vector<Triangle>* triangles, size_t total_vertex_count) {
    std::vector<TriangleMesh*> result;
    std::vector<int> group_id(segment_union.size(), -1); // 특정 요소가 속한 그룹 id.

    int group_index = 0;
    for (int i = 0; i < segment_union.size(); i++) {
        int group_root = segment_union[i];
        int& g_id = group_id[group_root];

        if (g_id == -1) {
            result.push_back(new TriangleMesh);
            g_id = group_index++;
            result[g_id]->material = new Material;
        }

        group_id[i] = g_id;
    }

    // 각 그룹을 하나의 블록이 담당.
    for (int g_id = 0; g_id < group_index; g_id++) {
        std::vector<int> index_lookup(total_vertex_count, -1);

        // 블록의 쓰레드가 각 요소를 돌며 연산 수행.
        for (int i = 0; i < segment_union.size(); i++) {
            if (group_id[i] != g_id)
                continue;

            glm::ivec3 new_index;
            for (int j = 0; j < 3; j++) {
                int& index_if_exist = index_lookup[triangles->at(i).id[j]];

                if (index_if_exist == -1) {
                    result[g_id]->vertex.push_back(triangles->at(i).vertex[j]);
                    index_if_exist = result[g_id]->vertex.size();
                }

                new_index[j] = index_if_exist;
            }

            result[g_id]->index.push_back(new_index);
        }
    }

    return result;
}
