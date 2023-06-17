#ifndef __FACEGRAPHUTILS_H
#define __FACEGRAPHUTILS_H

#include "glm/vector_relational.hpp"
#include "triangle.h"
#include "trianglemesh.hpp"
#include <unordered_map>
#include <vector>

struct Vec3Hash {
    std::size_t operator()(const glm::vec3& v) const;
};

bool is_connected(const Triangle& a, const Triangle& b);

TriangleMesh* triangle_list_to_obj(const std::vector<Triangle>& list);

#endif
