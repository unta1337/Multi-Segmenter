#ifndef __SERIALSEGMENTOR_H
#define __SERIALSEGMENTOR_H

#include "color.hpp"
#include "facegraph.h"
#include "model.hpp"
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

class SerialSegmenter : public Segmenter {
  public:
    SerialSegmenter(TriangleMesh* mesh, float tolerance = 0.0f);
    virtual std::vector<TriangleMesh*> do_segmentation();
};

#endif
