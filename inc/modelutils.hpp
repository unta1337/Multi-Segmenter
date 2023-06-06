#ifndef __MODELUTILS_H
#define __MODELUTILS_H

#include "trianglemesh.hpp"
#include <tiny_obj_loader.h>

int add_vertex(TriangleMesh* mesh, tinyobj::attrib_t& attributes, const tinyobj::index_t& idx,
               std::map<int, int>& known_vertices) {
    if (known_vertices.find(idx.vertex_index) != known_vertices.end())
        return known_vertices[idx.vertex_index];

    const glm::vec3* vertex_array = (const glm::vec3*)attributes.vertices.data();
    const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
    const glm::vec2* texcoord_array = (const glm::vec2*)attributes.texcoords.data();

    int new_id = (int)mesh->vertex.size();
    known_vertices[idx.vertex_index] = new_id;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);

    return new_id;
}

#endif
