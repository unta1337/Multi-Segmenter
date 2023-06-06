#ifndef __FACEGRAPH_H
#define __FACEGRAPH_H

#include "dstimer.hpp"
#include "facegraphutils.h"
#include "triangle.h"
#include "trianglemesh.hpp"
#include <stack>
#include <unordered_map>
#include <vector>

/**
 * @brief 삼각형에 대한 인접 리스트를 유지.
 */
class FaceGraph {
public:
    /**
     * 삼각형에 대한 인접 리스트
     */
    std::vector<std::vector<int>> adj_triangles;
    /**
     * 면을 인덱스로 다루기 위한 실제 면 정보 모음.
     */
    std::vector<Triangle>* triangles;

    DS_timer* timer;

    FaceGraph(std::vector<Triangle>* triangles, DS_timer* timer) : triangles(triangles), timer(timer) {
    }

    FaceGraph(std::vector<Triangle>* triangles) : FaceGraph(triangles, new DS_timer(16)) {
    }

    virtual ~FaceGraph() {
        if (timer->getNumCounter() == 16) {
            delete timer;
        }
    }

    //virtual void init() = 0;

    /**
     * @brief 인접 리스트에 있는 면을 인접한 면끼리 분류.
     * @details adj_list는 면의 법선 벡터를 기준으로 분류된 정보.
     * 이를 다시 인접한 면끼리 분류하여 세그멘테이션 수행.
     * @returns 인접한 면끼리 분류된 배열.
     */
    virtual std::vector<std::vector<Triangle>> get_segments() = 0;

    /**
     * @brief 인접 리스트에 있는 면을 순회하며 인덱스 부여.
     * @param visit 방문 추적용 배열.
     * @param start_vert 순회를 시작할 정점 인덱스.
     * @param count 순회할 정점에 부여될 인덱스.
     */
    virtual void traverse_dfs(std::vector<int>& visit, int start_vert, int count) = 0;

    void init() {
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
};
#endif
