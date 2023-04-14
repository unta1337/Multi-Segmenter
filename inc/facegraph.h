#ifndef __FACEGRAPH_H
#define __FACEGRAPH_H

#include "facegraphutils.h"
#include "glm/vec3.hpp"
#include "glm/vector_relational.hpp"
#include "triangle.h"
#include "trianglemesh.hpp"
#include <stack>
#include <unordered_map>
#include <vector>

namespace FaceGraph {
class FaceGraph {
  public:
    std::vector<std::vector<int>> adj_list;
    std::vector<Triangle>* ref_vector;

    FaceGraph(std::vector<Triangle>* list);
    std::vector<std::vector<Triangle>> check_connected();
    void traverse_dfs(std::vector<int>* visit, int start_vert, int count);
};
} // namespace FaceGraph
#endif
