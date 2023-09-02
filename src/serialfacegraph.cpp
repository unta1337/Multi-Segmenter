#include "serialfacegraph.h"

SerialFaceGraph::SerialFaceGraph(std::vector<Triangle>* triangles, DS_timer* timer) : FaceGraph(triangles, timer) {
    init();
}

SerialFaceGraph::SerialFaceGraph(std::vector<Triangle>* triangles) : FaceGraph(triangles) {
    init();
}

void SerialFaceGraph::init() {
    timer->onTimer(TIMER_FACEGRAPH_INIT_A);
    // 정점 -> 정점과 인접한 삼각형 매핑.
    std::unordered_map<glm::vec3, std::vector<int>, Vec3Hash> vertex_adjacent_map;
    for (int i = 0; i < triangles->size(); i++) {
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = triangles->at(i).vertex[j];
            vertex_adjacent_map[vertex].push_back(i);
        }
    }
    timer->offTimer(TIMER_FACEGRAPH_INIT_A);

    timer->onTimer(TIMER_FACEGRAPH_INIT_B);
    // 각 면에 대한 인접 리스트 생성.
    adj_triangles = std::vector<std::vector<int>>(triangles->size());

    // 각 삼각형에 대해서,
    for (int i = 0; i < triangles->size(); i++) {
        // 그 삼각형에 속한 정점과,
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = triangles->at(i).vertex[j];
            std::vector<int> adjacent_triangles = vertex_adjacent_map[vertex];
            // 맞닿아 있는 삼각형이,
            for (int k = 0; k < adjacent_triangles.size(); k++) {
                int adjacent_triangle = adjacent_triangles[k];

                // 자기 자신이 아니고,
                // 원래의 삼각형과도 맞닿아 있으면 인접 리스트에 추가.
                if (i != adjacent_triangle && is_connected(triangles->at(i), triangles->at(adjacent_triangle))) {
                    adj_triangles[i].push_back(adjacent_triangle);
                }
            }
        }
    }
    timer->offTimer(TIMER_FACEGRAPH_INIT_B);
}

std::vector<std::vector<Triangle>> SerialFaceGraph::get_segments() {
    timer->onTimer(TIMER_FACEGRAPH_GET_SETMENTS_A);

    std::vector<int> is_visit(adj_triangles.size());
    // 방문했다면 정점이 속한 그룹의 카운트 + 1.

    int count = 0;
    for (int i = 0; i < adj_triangles.size(); i++) {
        if (is_visit[i] == 0) {
            traverse_dfs(is_visit, i, ++count);
        }
    }
    timer->offTimer(TIMER_FACEGRAPH_GET_SETMENTS_A);

    timer->onTimer(TIMER_FACEGRAPH_GET_SETMENTS_B);
    std::vector<std::vector<Triangle>> component_list(count);

    for (int i = 0; i < is_visit.size(); i++) {
        component_list[is_visit[i] - 1].push_back(triangles->data()[i]);
    }

    timer->offTimer(TIMER_FACEGRAPH_GET_SETMENTS_B);

    return component_list;
}

void SerialFaceGraph::traverse_dfs(std::vector<int>& visit, int start_vert, int count) {
    std::stack<int> dfs_stack;
    dfs_stack.push(start_vert);

    while (!dfs_stack.empty()) {
        int current_vert = dfs_stack.top();
        dfs_stack.pop();

        visit[current_vert] = count;
        for (int i = 0; i < adj_triangles[current_vert].size(); i++) {
            int adjacent_triangle = adj_triangles[current_vert][i];
            if (visit[adjacent_triangle] == 0) {
                dfs_stack.push(adjacent_triangle);
            }
        }
    }
}
