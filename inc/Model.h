#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <string>

class Material_t {
public:
    std::string name;
    glm::vec3 ambient; //Ka
    glm::vec3 diffuse; //Kd
    glm::vec3 specular; //Ks
    glm::vec3 emmision; // Ke
    float shininess; //
    float opticalDensity; // Ni
    float dissolve; // d
    int illuminationModel; // illum
};

class TriangleMesh {
public:
    ~TriangleMesh(){
        delete material;
    }
    std::string name;
    std::vector<glm::vec3> vertex;
    std::vector<glm::vec3> normal;
    std::vector<glm::vec2> texcoord;
    std::vector<glm::ivec3> index;

    Material_t *material;
    int materialTextrueID{ -1 };
};

struct Texture {
    ~Texture()
    {
        if (pixel) delete[] pixel;
    }

    uint32_t* pixel{ nullptr };
    glm::ivec2     resolution{ -1 };
};

struct Model {
    ~Model()
    {
        for (auto mesh : meshes) delete mesh;
        for (auto texture : textures) delete texture;
    }

    std::vector<TriangleMesh*> meshes;
    std::vector<Texture*>      textures;

    void readOBJ(std::string objFilePath);
};