#ifndef __MATERIAL_H
#define __MATERIAL_H

#include <string>
#include "vector.h"

/**
 * @brief mtl 파일에 대한 자료형.
 * @details .obj 파일을 내보내기 위한 내부 자료형.
 */
class Material {
  public:
    std::string name;
    Vector3f ambient;      // Ka
    Vector3f diffuse;      // Kd
    Vector3f specular;     // Ks
    Vector3f emission;     // Ke
    float shininess;        // Ns
    float dissolve;         // d
    int illumination_model; // illum
};
#endif
