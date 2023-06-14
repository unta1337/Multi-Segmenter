#ifndef __MATERIAL_H
#define __MATERIAL_H

#include <glm/glm.hpp>
#include <string>

/**
 * @brief mtl 파일에 대한 자료형.
 * @details .obj 파일을 내보내기 위한 내부 자료형.
 */
class Material {
  public:
    char name[255];
    glm::vec3 ambient;      // Ka
    glm::vec3 diffuse;      // Kd
    glm::vec3 specular;     // Ks
    glm::vec3 emission;     // Ke
    float shininess;        // Ns
    float dissolve;         // d
    int illumination_model; // illum
};
#endif
