#ifndef __COLOR_H
#define __COLOR_H

#include "vector.h"
#include <algorithm>

namespace Color {
inline Vector4f get_color_from_jet(float v, float vmin, float vmax) {
    Vector4f c = {1.0, 1.0, 1.0, 1.0};
    float dv;

    v = std::clamp(v, vmin, vmax);
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c.x = 0;
        c.y = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c.x = 0;
        c.z = 1 + (float)(4 * (vmin + 0.25 * dv - v) / dv);
    } else if (v < (vmin + 0.75 * dv)) {
        c.x = 4 * (float)((v - vmin - 0.5 * dv) / dv);
        c.z = 0;
    } else {
        c.y = 1 + (float)(4 * (vmin + 0.75 * dv - v) / dv);
        c.z = 0;
    }
    return c;
}
} // namespace Color
#endif
