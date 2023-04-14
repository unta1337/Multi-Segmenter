#ifndef __SERIALSEGMENTOR_H
#define __SERIALSEGMENTOR_H

#include "color.hpp"
#include "facegraph.hpp"
#include "model.hpp"
#include "segmentor.hpp"
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

class SerialSegmentor : public Segmentor {
  public:
    SerialSegmentor(TriangleMesh* mesh, float tolerance = 0.0f);
    virtual std::vector<TriangleMesh*> do_segmentation();
};

#endif
