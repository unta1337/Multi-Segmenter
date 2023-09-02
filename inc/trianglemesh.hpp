#ifndef __TRIANGLEMESH_H
#define __TRIANGLEMESH_H

#include "material.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>

/**
 * @brief 오브젝트에서 하나의 속성을 공유하는 삼각형의 집합.
 * @details 최초 입력된 오브젝트의 경우 세그멘테이션이 진행되기 전이므로 하나의
 * 집합으로 구성된다.
 * 세그멘테이션이 완료되면 오브젝트의 부분별로 나뉘어 각각이 하나의 집합이
 * 된다.
 */
class TriangleMesh {
  public:
    ~TriangleMesh() {
        delete material;
    }
    /**
     * 그룹 이름.
     */
    char name[255];
    /**
     * 그룹에 속한 정점 목록.
     */
    std::vector<glm::vec3> vertex;
    /**
     * 그룹에 속한 정점의 법선 벡터.
     */
    std::vector<glm::vec3> normal;
    /**
     * 그룹에 속한 정점의 텍스쳐 좌표 정보.
     */
    std::vector<glm::vec2> texcoord;
    /**
     * 그룹에 속한 정점들로 이뤄지는 면에 대한 정보.
     */
    std::vector<glm::ivec3> index;

    /**
     * 그룹에 일괄적으로 적용되는 재질.
     * 오브젝트의 각 부분을 색상 등으로 구분하기 위해 사용.
     */
    Material* material;

    /**
     * 그룹에 적용되는 재질에 대응하는 인덱스.
     */
    int material_texture_id{-1};
};

#endif
