#ifndef __COLOR_H
#define __COLOR_H

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

namespace Color {
inline glm::vec4 get_color_from_jet(float v, float vmin, float vmax) {
    glm::vec3 c = {1.0, 1.0, 1.0};
    float dv;

    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c.r = 0;
        c.g = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c.r = 0;
        c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c.r = 4 * (v - vmin - 0.5 * dv) / dv;
        c.b = 0;
    } else {
        c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c.b = 0;
    }
    return glm::vec4(c, 1.0f);
}
} // namespace Color
#endif
