#include "Model.h"
#include "Segmentation.h"
#include <fstream>
#include <iostream>

bool writeOBJ(std::vector<TriangleMesh*> meshes, std::string path,
              bool isOneFile);

int main() {
    std::cout << "Hello world!" << std::endl;

    Model tmp;
    tmp.readOBJ("assets/Cube_noised.obj");

    std::cout << "mesh 1 vertex size : " << tmp.meshes[0]->vertex.size()
              << std::endl;
    std::cout << "mesh 1 triangle size : " << tmp.meshes[0]->index.size()
              << std::endl;

    Segmentation_t segs(tmp.meshes[0], 15.f);
    auto seg = segs.DoSegmentation();

    std::cout << "seg result " << seg.size() << std::endl;
    for (auto& s : seg) {
        std::cout << s->name << std::endl;
        std::cout << s->vertex.size() << std::endl;
    }

    writeOBJ(seg, "", false);
    writeOBJ(seg, "Segmented_cube", true);

    return 0;
}

// path use only isOneFile is true
// And also path must be write without extensions (.ply, .mtl)
bool writeOBJ(std::vector<TriangleMesh*> meshes, std::string path,
              bool isOneFile) {

    if (isOneFile) {
        std::ofstream mtlFile(path + ".mtl");

        if (!mtlFile.is_open())
            return false;

        for (int i = 0; i < meshes.size(); i++) {
            TriangleMesh* mesh = meshes[i];

            mtlFile << "newmtl " << mesh->material->name << std::endl;
            mtlFile << "Ka " << mesh->material->ambient.x << " "
                    << mesh->material->ambient.y << " "
                    << mesh->material->ambient.z << std::endl;
            mtlFile << "Kd " << mesh->material->diffuse.x << " "
                    << mesh->material->diffuse.y << " "
                    << mesh->material->diffuse.z << std::endl;
            mtlFile << "Ks " << mesh->material->specular.x << " "
                    << mesh->material->specular.y << " "
                    << mesh->material->specular.z << std::endl;
            mtlFile << "d " << std::to_string(1.000000f) << std::endl;
            mtlFile << "illum 2" << std::endl;
            mtlFile << std::endl;
        }
        mtlFile.close();

        std::ofstream objFile(path + ".obj");
        if (!objFile.is_open())
            return false;

        size_t vertexLength = 0;
        objFile << "mtllib " << path << ".mtl" << std::endl;
        for (int i = 0; i < meshes.size(); i++) {
            TriangleMesh* mesh = meshes[i];

            objFile << "o " << mesh->name << std::endl;

            for (auto v : mesh->vertex) {
                objFile << "v " << v.x << " " << v.y << " " << v.z << std::endl;
            }

            objFile << "usemtl " << mesh->material->name << std::endl;
            objFile << "s off" << std::endl;

            for (int i = 0; i < mesh->index.size(); i++) {
                objFile << "f " << vertexLength + mesh->index[i].x << " "
                        << vertexLength + mesh->index[i].y << " "
                        << vertexLength + mesh->index[i].z << std::endl;
            }

            vertexLength += mesh->vertex.size();
        }
        objFile.close();
    } else {
        for (int i = 0; i < meshes.size(); i++) {
            TriangleMesh* mesh = meshes[i];
            std::ofstream mtlFile(mesh->material->name + ".mtl");

            if (!mtlFile.is_open())
                return false;

            mtlFile << "newmtl " << mesh->material->name << std::endl;
            mtlFile << "Ka " << mesh->material->ambient.x << " "
                    << mesh->material->ambient.y << " "
                    << mesh->material->ambient.z << std::endl;
            mtlFile << "Kd " << mesh->material->diffuse.x << " "
                    << mesh->material->diffuse.y << " "
                    << mesh->material->diffuse.z << std::endl;
            mtlFile << "Ks " << mesh->material->specular.x << " "
                    << mesh->material->specular.y << " "
                    << mesh->material->specular.z << std::endl;
            mtlFile << "d " << std::to_string(1.000000f) << std::endl;
            mtlFile << "illum 2" << std::endl;

            mtlFile.close();

            std::ofstream objFile(mesh->name + ".obj");

            if (!objFile.is_open())
                return false;

            objFile << "mtllib " << mesh->material->name << ".mtl" << std::endl;

            objFile << "o " << mesh->name << std::endl;

            for (auto v : mesh->vertex) {
                objFile << "v " << v.x << " " << v.y << " " << v.z << std::endl;
            }

            objFile << "usemtl " << mesh->material->name << std::endl;
            objFile << "s off" << std::endl;

            for (int i = 0; i < mesh->index.size(); i++) {
                objFile << "f " << mesh->index[i].x << " " << mesh->index[i].y
                        << " " << mesh->index[i].z << std::endl;
            }

            objFile.close();
        }
    }

    return true;
}
