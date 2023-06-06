#ifndef __MODEL_H
#define __MODEL_H

#include "material.h"
#include "texture.hpp"
#include "trianglemesh.hpp"
#include <glm/glm.hpp>
#include <iostream>
#include <set>
#include <string>
#include <tiny_obj_loader.h>
#include <vector>

/**
 * @brief 하나의 .obj 파일을 저장하는 내부 자료형.
 */
struct Model {
    ~Model();

    /**
     * 오브젝트에 포함된 메시 그룹 모음.
     */
    std::vector<TriangleMesh*> meshes;
    /**
     * 오브젝트에 사용된 텍스쳐 모음.
     */
    std::vector<Texture*> textures;

    /**
     * @brief 경로에 있는 .obj 파일 로드.
     * @param objFilePath 오브젝트 파일 경로.
     */
    void read_obj(std::string obj_file_path);
};

#endif
