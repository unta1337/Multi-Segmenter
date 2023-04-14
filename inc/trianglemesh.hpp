#ifndef __TRIANGLEMESH_H
#define __TRIANGLEMESH_H
#include "material.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>

class TriangleMesh {
  public:
    ~TriangleMesh() {
        delete material;
    }
    std::string name;
    std::vector<glm::vec3> vertex;
    std::vector<glm::vec3> normal;
    std::vector<glm::vec2> texcoord;
    std::vector<glm::ivec3> index;

    Material* material;
    int materialTextureID{-1};
};

#endif
