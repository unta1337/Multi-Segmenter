#ifndef __CUDAFACEGRAPHUTILS_H
#define __CUDAFACEGRAPHUTILS_H

#include "segmentunion.hpp"
#include "triangle.h"
#include "trianglemesh.hpp"
#include <algorithm>
#include <vector>

std::vector<TriangleMesh*> segment_union_to_obj(const SegmentUnion segment_union,
                                                const std::vector<Triangle>* triangles, size_t total_vertex_count);

#endif
