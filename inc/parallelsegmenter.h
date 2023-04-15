#ifndef __PARALLELSEGMENTER_H
#define __PARALLELSEGMENTER_H

#include "segmenter.hpp"
#include "trianglemesh.hpp"
#include <serialfacegraph.h> // parallel로 변경
#include <unordered_map>
#include <vector>

class ParallelSegmenter : public Segmenter {
  public:
    ParallelSegmenter(TriangleMesh* mesh, float tolerance = 0.0f);
    virtual std::vector<TriangleMesh*> do_segmentation();
    inline glm::vec3 get_normal_key(std::unordered_map<glm::vec3, size_t, Vec3Hash>& count_map, glm::vec3& normal);
    inline void init_count_map(std::unordered_map<glm::vec3, size_t, Vec3Hash>& count_map,
                               std::vector<glm::vec3>& face_normals);
};

#endif
