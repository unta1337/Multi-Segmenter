#ifndef __MATERIAL_H
#define __MATERIAL_H

#include <glm/glm.hpp>
#include <string>

class Material {
  public:
    std::string name;
    glm::vec3 ambient;     // Ka
    glm::vec3 diffuse;     // Kd
    glm::vec3 specular;    // Ks
    glm::vec3 emission;    // Ke
    float shininess;       // Ns
    float optical_density;  // Ni
    float dissolve;        // d
    int illumination_model; // illum
};
#endif
