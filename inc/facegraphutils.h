#ifndef __FACEGRAPHUTILS_H
#define __FACEGRAPHUTILS_H

#include "glm/vector_relational.hpp"
#include "triangle.h"
#include "trianglemesh.hpp"
#include <unordered_map>
#include <vector>

namespace FaceGraph {
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

inline bool is_connected(Triangle a, Triangle b) {
    int v = 0;
    if (glm::all(glm::equal(a.vertex[0], b.vertex[0])) ||
        glm::all(glm::equal(a.vertex[0], b.vertex[1])) ||
        glm::all(glm::equal(a.vertex[0], b.vertex[2])))
        v++;
    if (glm::all(glm::equal(a.vertex[1], b.vertex[0])) ||
        glm::all(glm::equal(a.vertex[1], b.vertex[1])) ||
        glm::all(glm::equal(a.vertex[1], b.vertex[2])))
        v++;
    if (glm::all(glm::equal(a.vertex[2], b.vertex[0])) ||
        glm::all(glm::equal(a.vertex[2], b.vertex[1])) ||
        glm::all(glm::equal(a.vertex[2], b.vertex[2])))
        v++;

    return (v > 1);
}

inline TriangleMesh* triangle_list_to_obj(std::vector<Triangle> list) {
    TriangleMesh* sub_object = new TriangleMesh();

    Material* sub_mtl = new Material();

    std::unordered_map<glm::vec3, size_t, Vec3Hash> vertex_map;
    sub_object->index.resize(list.size());
    size_t vert_idx = 1;
    for (int i = 0; i < list.size(); i++) {
        glm::ivec3 index;
        for (int j = 0; j < 3; j++) {
            auto vertex_item = vertex_map.find(list[i].vertex[j]);

            if (vertex_item != vertex_map.end()) {
                index[j] = (int) vertex_item->second;
            } else {
                vertex_map.insert({list[i].vertex[j], vert_idx});
                index[j] = (int) vert_idx++;
            }
            // auto vertIter = std::find(sub_object->vertex.begin(),
            // sub_object->vertex.end(), list[i].vertex[j]);

            ////찾은경우
            // if (vertIter != sub_object->vertex.end()) {
            //     index[j] = vertIter - sub_object->vertex.begin() + 1;
            // }
            ////못찾은경우
            // else {
            //     sub_object->vertex.push_back(list[i].vertex[j]);
            //     index[j] = sub_object->vertex.end() -
            //     sub_object->vertex.begin();
            // }
        }
        sub_object->index[i] = index;
    }

    sub_object->vertex.resize(vert_idx);
    for (auto v_item : vertex_map) {
        sub_object->vertex[v_item.second - 1] = v_item.first;
    }

    sub_object->material = sub_mtl;

    return sub_object;
}
} // namespace FaceGraph
#endif
