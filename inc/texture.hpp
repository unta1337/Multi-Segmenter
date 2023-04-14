#ifndef __TEXTURE_H
#define __TEXTURE_H

#include "glm/ext/vector_int2.hpp"
#include <cstdint>

struct Texture {
    ~Texture() {
        if (pixel)
            delete[] pixel;
    }

    uint32_t* pixel{nullptr};
    glm::ivec2 resolution{-1};
};
#endif
