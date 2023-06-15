#ifndef __CUDASEGMENTER_H
#define __CUDASEGMENTER_H

#include "color.hpp"
#include "devicetrianglemesh.hpp"
#include "thrustlib.h"
#include "dstimer.hpp"
#include "facegraph.hpp"
#include "logutils.h"
#include "model.h"
#include "segmenter.hpp"
#include "serialfacegraph.h"
#include "trianglemesh.hpp"
#include <algorithm>
#include <chrono>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/normal.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <vector>

class CUDASegmenter : public Segmenter {
  public:
    CUDASegmenter(TriangleMesh* mesh, float tolerance = 0.0f);
    ~CUDASegmenter();
    virtual std::vector<TriangleMesh*> do_segmentation();
    DeviceTriangleMesh * deviceMesh;
};
#endif
