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

struct Model {
    ~Model() {
        for (auto mesh : meshes)
            delete mesh;
        for (auto texture : textures)
            delete texture;
    }

    std::vector<TriangleMesh*> meshes;
    std::vector<Texture*> textures;

    void read_obj(std::string obj_file_path);
};

#endif
