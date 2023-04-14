#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include "glm/vec3.hpp"

namespace FaceGraph {
typedef struct Triangle {
    glm::vec3 vertex[3];
} Triangle;
} // namespace FaceGraph

#endif
