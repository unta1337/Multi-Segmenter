#ifndef __OBJ_UTILS_H
#define __OBJ_UTILS_H

#include "trianglemesh.hpp"
#include <fstream>
#include <vector>

// path use only is_one_file is true
// And also path must be write without extensions (.ply, .mtl)
inline bool write_obj(std::vector<TriangleMesh*> meshes, std::string path, bool is_one_file) {
    // 참고:
    // TriangleMesh 자체가 .obj 포맷으로 구성되어 있음.
    // name: 각 부분의 명칭 -> "o cube_seg0"
    // vertex: 정점 정보 -> "v 0.5 0.7 1.45"
    // normal: 정점의 법선 벡터 -> 사용하지 않음.
    // texcoord: 텍스쳐 매핑 -> 사용하지 않음.
    // index: 그룹에 속한 면 정보 -> "f 153 29 598"
    // material, materialTextureID: 재질 정보 -> "mtllib Segmented_cube.mtl"

    if (is_one_file) {
        std::ofstream mtl_file(path + ".mtl");

        if (!mtl_file.is_open())
            return false;

        for (int i = 0; i < meshes.size(); i++) {
            TriangleMesh* mesh = meshes[i];

            mtl_file << "newmtl " << mesh->material->name << std::endl;
            mtl_file << "Ka " << mesh->material->ambient.x << " " << mesh->material->ambient.y << " "
                     << mesh->material->ambient.z << std::endl;
            mtl_file << "Kd " << mesh->material->diffuse.x << " " << mesh->material->diffuse.y << " "
                     << mesh->material->diffuse.z << std::endl;
            mtl_file << "Ks " << mesh->material->specular.x << " " << mesh->material->specular.y << " "
                     << mesh->material->specular.z << std::endl;
            mtl_file << "d " << std::to_string(1.000000f) << std::endl;
            mtl_file << "illum 2" << std::endl;
            mtl_file << std::endl;
        }
        mtl_file.close();

        std::ofstream obj_file(path + ".obj");
        if (!obj_file.is_open())
            return false;

        size_t vertex_length = 0;
        obj_file << "mtllib " << path << ".mtl" << std::endl;
        for (int i = 0; i < meshes.size(); i++) {
            TriangleMesh* mesh = meshes[i];

            obj_file << "o " << mesh->name << std::endl;

            for (auto v : mesh->vertex) {
                obj_file << "v " << v.x << " " << v.y << " " << v.z << std::endl;
            }

            obj_file << "usemtl " << mesh->material->name << std::endl;
            obj_file << "s off" << std::endl;

            for (int i = 0; i < mesh->index.size(); i++) {
                obj_file << "f " << vertex_length + mesh->index[i].x << " " << vertex_length + mesh->index[i].y << " "
                         << vertex_length + mesh->index[i].z << std::endl;
            }

            vertex_length += mesh->vertex.size();
        }
        obj_file.close();
    } else {
        for (int i = 0; i < meshes.size(); i++) {
            TriangleMesh* mesh = meshes[i];
            std::ofstream mtl_file(path + mesh->material->name + ".mtl");

            if (!mtl_file.is_open())
                return false;

            mtl_file << "newmtl " << mesh->material->name << std::endl;
            mtl_file << "Ka " << mesh->material->ambient.x << " " << mesh->material->ambient.y << " "
                     << mesh->material->ambient.z << std::endl;
            mtl_file << "Kd " << mesh->material->diffuse.x << " " << mesh->material->diffuse.y << " "
                     << mesh->material->diffuse.z << std::endl;
            mtl_file << "Ks " << mesh->material->specular.x << " " << mesh->material->specular.y << " "
                     << mesh->material->specular.z << std::endl;
            mtl_file << "d " << std::to_string(1.000000f) << std::endl;
            mtl_file << "illum 2" << std::endl;

            mtl_file.close();

            std::ofstream obj_file(path + mesh->name + ".obj");

            if (!obj_file.is_open())
                return false;

            obj_file << "mtllib " << mesh->material->name << ".mtl" << std::endl;

            obj_file << "o " << mesh->name << std::endl;

            for (auto v : mesh->vertex) {
                obj_file << "v " << v.x << " " << v.y << " " << v.z << std::endl;
            }

            obj_file << "usemtl " << mesh->material->name << std::endl;
            obj_file << "s off" << std::endl;

            for (int i = 0; i < mesh->index.size(); i++) {
                obj_file << "f " << mesh->index[i].x << " " << mesh->index[i].y << " " << mesh->index[i].z << std::endl;
            }

            obj_file.close();
        }
    }

    return true;
}

#endif
