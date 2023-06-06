#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include "vector.h"

/**
 * @brief 세 개의 정점으로 정의하는 삼각형 자료형.
 */
struct Triangle {
    /**
     * 삼각형을 구성하는 세 개의 정점 좌표.
     */
    Vector3f vertex[3];
};

#endif
