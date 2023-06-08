#ifndef __CUDAFACEGRAPHUTILS_H
#define __CUDAFACEGRAPHUTILS_H

#include "triangle.h"
#include "trianglemesh.hpp"
#include <vector>

std::vector<TriangleMesh*> segment_union_to_obj(const std::vector<int> segment_union,
                                                const std::vector<Triangle>* triangles, size_t total_vertex_count);

#endif
