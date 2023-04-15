#ifndef __ORIGINALSEGMENTER_H
#define __ORIGINALSEGMENTER_H

#include "color.hpp"
#include "facegraph.hpp"
#include "model.h"
#include "segmenter.hpp"
#include "trianglemesh.hpp"
#include <algorithm>
#include <chrono>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/normal.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <vector>

class OriginalSegmenter : public Segmenter {
  public:
    OriginalSegmenter(TriangleMesh* mesh, float tolerance = 0.0f);
    virtual std::vector<TriangleMesh*> do_segmentation();
};

#endif
