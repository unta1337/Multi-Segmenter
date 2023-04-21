#include "parallelfacegraph.h"
#include "lockutils.hpp"

ParallelFaceGraph::ParallelFaceGraph(std::vector<Triangle>* triangles, DS_timer* timer) : FaceGraph(triangles, timer) {
    init();
}

ParallelFaceGraph::ParallelFaceGraph(std::vector<Triangle>* triangles) : FaceGraph(triangles) {
    init();
}

struct AdjacentNode {
    glm::vec3* vertex = nullptr;
    int* adjacents;
    int filled_index = 0;
} typedef AdjacentNode;

void ParallelFaceGraph::init() {
    timer->onTimer(TIMER_FACEGRAPH_INIT_A);

    /* 변수 선언 */
    Vec3Hash hash_function;
    size_t vertex_size = triangles->size() * 3;
    omp_lock_t* locks = new_locks(vertex_size);
    // 해제 주의
    size_t** vertex_hash_list;

    /* 초기화 */
    vertex_hash_list = new size_t*[vertex_size];
    for (int i = 0; i < vertex_size; i++) {
        vertex_hash_list[i] = new size_t[3];
    }

    /* 해시 리스트 초기화 및 카운트 세기 */
    int* over_count_map = new int[vertex_size];
    std::fill_n(over_count_map, vertex_size, 0);
    // duplicated_vertex_key에 중복 vertex들의 triangle 포함 횟수 합산

    #pragma omp parallel for
    for (int i = 0; i < triangles->size(); i++) {
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = triangles->at(i).vertex[j];
            size_t index = vertex_hash_list[i][j] = hash_function(vertex) % vertex_size;
            #pragma omp atomic
            over_count_map[index]++;
        }
    }

    int max_count = 0;
    // #pragma omp parallel for reduction(max:max_count)
    for (int i = 0; i < vertex_size; i++) {
        if (over_count_map[i] > max_count) {
            max_count = over_count_map[i];
        }
    }

    AdjacentNode* adjacent_nodes = new AdjacentNode[vertex_size];
    #pragma omp parallel for
    for (int i = 0; i < triangles->size(); i++) {
        for (int j = 0; j < 3; j++) {
            glm::vec3& vertex = triangles->at(i).vertex[j];
            size_t vertex_hash = vertex_hash_list[i][j];
            bool is_exist = false;
            AdjacentNode* node = &adjacent_nodes[vertex_hash];

            size_t locked_hash = vertex_hash;
            omp_set_lock(&locks[locked_hash]);
            while (node->vertex != nullptr) {
                glm::vec3* target = node->vertex;
                if (vertex == *target) {
                    is_exist = true;
                    break;
                }
                vertex_hash = (vertex_hash + 1) % vertex_size;
                node = &adjacent_nodes[vertex_hash];
            }
            vertex_hash_list[i][j] = vertex_hash;
            omp_unset_lock(&locks[locked_hash]);

            omp_set_lock(&locks[vertex_hash]);
            if (!is_exist) {
                node->vertex = &vertex;
                node->adjacents = new int[max_count];
                std::fill_n(node->adjacents, max_count, false);
            }
            node->adjacents[node->filled_index++] = i;
            omp_unset_lock(&locks[vertex_hash]);
        }
    }

    timer->offTimer(TIMER_FACEGRAPH_INIT_A);

    timer->onTimer(TIMER_FACEGRAPH_INIT_B);
    // 각 면에 대한 인접 리스트 생성.
    adj_triangles = std::vector<std::vector<int>>(triangles->size());

    // 각 삼각형에 대해서,
    #pragma omp parallel for
    for (int i = 0; i < triangles->size(); i++) {
        // 그 삼각형에 속한 정점과,
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = triangles->at(i).vertex[j];

            size_t vertex_hash = vertex_hash_list[i][j];
            AdjacentNode* node = &adjacent_nodes[vertex_hash];

            int* adjacents = node->adjacents;

            // 맞닿아 있는 삼각형이,
            for (int k = 0; k < node->filled_index; k++) {
                int adjacent_triangle = adjacents[k];
                // 자기 자신이 아니고,
                // 원래의 삼각형과도 맞닿아 있으면 인접 리스트에 추가.
                if (i != adjacent_triangle && is_connected(triangles->at(i), triangles->at(adjacent_triangle))) {
                    LOCK(locks, i, adj_triangles[i].push_back(adjacent_triangle));
                }
            }
        }
    }

    for (int i = 0; i < vertex_size; i++) {
        delete[] vertex_hash_list[i];
    }
    delete[] vertex_hash_list;

    delete[] over_count_map;

    delete[] adjacent_nodes;

    destroy_locks(locks, vertex_size);
    timer->offTimer(TIMER_FACEGRAPH_INIT_B);
}
int value(int va) {
    return va;
}
std::vector<std::vector<Triangle>> ParallelFaceGraph::get_segments() {
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

void ParallelFaceGraph::traverse_dfs(std::vector<int>& visit, int start_vert, int count) {
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
