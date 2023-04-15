#ifndef __SERIALSEGMENTER_H
#define __SERIALSEGMENTER_H

#include "color.hpp"
#include "dstimer.h"
#include "facegraph.hpp"
#include "logutil.hpp"
#include "model.h"
#include "segmenter.hpp"
#include "trianglemesh.hpp"
#include <algorithm>
#include <chrono>
#include <format>
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

  private:
    inline glm::vec3 get_normal_key(std::unordered_map<glm::vec3, size_t, Vec3Hash>& count_map, glm::vec3& normal);

    inline void init_count_map(std::unordered_map<glm::vec3, size_t, Vec3Hash>& count_map,
                               std::vector<glm::vec3>& face_normals);
};
#endif
