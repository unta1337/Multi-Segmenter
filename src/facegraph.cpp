#include "facegraph.h"

namespace FaceGraph {
FaceGraph::FaceGraph(std::vector<Triangle>* list) {
    ref_vector = list;

    std::unordered_map<glm::vec3, std::vector<int>, Vec3Hash> vertex_map;
    for (int i = 0; i < list->size(); i++) {
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = list->at(i).vert[j];
            vertex_map[vertex].push_back(i);
        }
    }

    adj_list = std::vector<std::vector<int>>(list->size());
    for (int i = 0; i < list->size(); i++) {
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = list->at(i).vert[j];
            std::vector<int> adjacent_triangles = vertex_map[vertex];
            for (int k = 0; k < adjacent_triangles.size(); k++) {
                int adjacent_triangle = adjacent_triangles[k];
                if (i != adjacent_triangle &&
                    is_connected(list->at(i), list->at(adjacent_triangle))) {
                    adj_list[i].push_back(adjacent_triangle);
                }
            }
        }
    }
}

std::vector<std::vector<Triangle>> FaceGraph::check_connected() {
    std::vector<int> is_visit(adj_list.size());

    int count = 0;
    for (int i = 0; i < adj_list.size(); i++) {
        if (is_visit[i] == 0) {
            count++;
            traverse_dfs(&is_visit, i, count);
            // std::cout << "// ";
        }
    }

    // std::cout << "Component number : " << count<< std::endl;

    std::vector<std::vector<Triangle>> component_list(count);

    for (int i = 0; i < is_visit.size(); i++) {
        component_list[is_visit[i] - 1].push_back(ref_vector->data()[i]);
    }

    return component_list;
}

void FaceGraph::traverse_dfs(std::vector<int>* visit, int start_vert,
                             int count) {
    // std::cout << start_vert << " ";
    std::stack<int> dfs_stack;
    dfs_stack.push(start_vert);

    while (!dfs_stack.empty()) {
        int current_vert = dfs_stack.top();
        dfs_stack.pop();

        visit->data()[current_vert] = count;
        for (int i = 0; i < adj_list[current_vert].size(); i++) {
            int adjacent_triangle = adj_list[current_vert][i];
            if (visit->data()[adjacent_triangle] == 0) {
                dfs_stack.push(adjacent_triangle);
            }
        }
    }
    /*visit->data()[start_vert] = count;

    for (int i = 0; i < adj_list[start_vert].size(); i++) {
        int adjacent_triangle = adj_list[start_vert][i];
        if (visit->data()[adjacent_triangle] == 0) {
            traverse_dfs(visit, adjacent_triangle, count);
        }
    }*/
}
} // namespace FaceGraph
