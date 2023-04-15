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

    FaceGraph(std::vector<Triangle>* triangles) {
        this->triangles = triangles;
    }

    /**
     * @brief 인접 리스트에 있는 면을 인접한 면끼리 분류.
     * @details adj_list는 면의 법선 벡터를 기준으로 분류된 정보.
     * 이를 다시 인접한 면끼리 분류하여 세그멘테이션 수행.
     * @returns 인접한 면끼리 분류된 배열.
     */
    virtual std::vector<std::vector<Triangle>> check_connected() = 0;

    /**
     * @brief 인접 리스트에 있는 면을 순회하며 인덱스 부여.
     * @param visit 방문 추적용 배열.
     * @param start_vert 순회를 시작할 정점 인덱스.
     * @param count 순회할 정점에 부여될 인덱스.
     */
    virtual void traverse_dfs(std::vector<int>& visit, int start_vert, int count) = 0;
};
#endif
