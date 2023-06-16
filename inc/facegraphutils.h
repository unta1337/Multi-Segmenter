#ifndef __FACEGRAPHUTILS_H
#define __FACEGRAPHUTILS_H

#include "glm/vector_relational.hpp"
#include "triangle.h"
#include "trianglemesh.hpp"
#include <unordered_map>
#include <vector>

struct Vec3Hash {
    std::size_t operator()(const glm::vec3& v) const {
        // Use the std::hash function to hash the individual components of the
        // vector
        std::size_t h1 = std::hash<float>()(v.x);
        std::size_t h2 = std::hash<float>()(v.y);
        std::size_t h3 = std::hash<float>()(v.z);

        // Combine the individual component hashes into a single hash value
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

__host__ __device__ bool is_connected(const Triangle& a, const Triangle& b) {
    int shared_vertices = 0;

    for (auto i : a.vertex) {
        for (auto j : b.vertex) {
            if (glm::all(glm::equal(i, j))) {
                shared_vertices++;
                break;
            }
        }
    }

    return (shared_vertices > 1);
}

inline TriangleMesh* triangle_list_to_obj(const std::vector<Triangle>& list) {
    TriangleMesh* sub_object = new TriangleMesh();

    Material* sub_mtl = new Material();

    std::unordered_map<glm::vec3, size_t, Vec3Hash> vertex_map;
    sub_object->index.resize(list.size());
    size_t vertex_obj_index = 1;
    // .obj 파일 포맷에서 인덱스는 1부터 시작.

    // 모든 삼각형에 대해서,
    for (int i = 0; i < list.size(); i++) {
        glm::ivec3 index;

        // 삼각형을 이루는 정점이,
        // 이미 확인한 정점이라면 대응하는 obj 인덱스를,
        // 아니라면 새로운 obj 인덱스를 부여.
        for (int j = 0; j < 3; j++) {
            auto vertex_item = vertex_map.find(list[i].vertex[j]);

            if (vertex_item != vertex_map.end()) {
                index[j] = (int)vertex_item->second;
            } else {
                vertex_map.insert({list[i].vertex[j], vertex_obj_index});
                index[j] = (int)vertex_obj_index++;
            }
        }
        sub_object->index[i] = index;
    }

    // 정점이 obj 인덱스와 대응하도록 대입.
    sub_object->vertex.resize(vertex_obj_index);
    for (auto v_item : vertex_map) {
        sub_object->vertex[v_item.second - 1] = v_item.first;
    }

    sub_object->material = sub_mtl;

    return sub_object;
}
#endif
