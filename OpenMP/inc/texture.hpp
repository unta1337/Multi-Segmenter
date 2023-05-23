#ifndef __TEXTURE_H
#define __TEXTURE_H

#include "glm/ext/vector_int2.hpp"
#include <cstdint>

/**
 * @brief 형식적인 텍스쳐 자료형.
 */
struct Texture {
    ~Texture() {
        if (pixel)
            delete[] pixel;
    }

    /**
     * 텍스쳐 정보.
     * 참고: unsigned 32-bit 자료형
     * -> 0xFFFFFFFF
     * -> FF FF FF FF
     * 오른쪽부터 차례대로 RGBA를 256 단계로 나타냄.
     * (단, 왼쪽, 오른쪽은 구현에 따라 상이.)
     */
    uint32_t* pixel{nullptr};

    /**
     * 텍스쳐 해상도.
     */
    glm::ivec2 resolution{-1};
};
#endif
