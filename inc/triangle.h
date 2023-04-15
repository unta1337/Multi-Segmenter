#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include "glm/vec3.hpp"

namespace FaceGraph {
/**
 * @brief 세 개의 정점으로 정의하는 삼각형 자료형.
 */
typedef struct Triangle {
    /**
     * 삼각형을 구성하는 세 개의 정점 좌표.
     * 제안: std::array<glm::vec3, 3>으로 STL 사용.
     */
    glm::vec3 vertex[3];
} Triangle;
} // namespace FaceGraph

#endif
