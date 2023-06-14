#ifndef __DEVICETRIANGLEMESH_H
#define __DEVICETRIANGLEMESH_H

#include "material.h"
#include "trianglemesh.hpp"
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <thrust/device_vector.h>

/**
 * @brief 오브젝트에서 하나의 속성을 공유하는 삼각형의 집합. (Device 용)
 * @details 최초 입력된 오브젝트의 경우 세그멘테이션이 진행되기 전이므로 하나의
 * 집합으로 구성된다.
 * 세그멘테이션이 완료되면 오브젝트의 부분별로 나뉘어 각각이 하나의 집합이
 * 된다.
 */
class DeviceTriangleMesh {
  public:
    DeviceTriangleMesh(TriangleMesh* mesh) {
        vertex_device_vector = new thrust::device_vector<glm::vec3>();
        *vertex_device_vector = mesh->vertex;
        vertex = thrust::raw_pointer_cast(vertex_device_vector->data());

        normal_device_vector = new thrust::device_vector<glm::vec3>();
        *normal_device_vector = mesh->normal;
        normal = thrust::raw_pointer_cast(normal_device_vector->data());

        texcoord_device_vector = new thrust::device_vector<glm::vec2>();
        *texcoord_device_vector = mesh->texcoord;
        texcoord = thrust::raw_pointer_cast(texcoord_device_vector->data());

        index_device_vector = new thrust::device_vector<glm::ivec3>();
        *index_device_vector = mesh->index;
        index = thrust::raw_pointer_cast(index_device_vector->data());

        cudaMalloc((void **) &name, sizeof(char) * 255);
        cudaMemcpy(name, mesh->name, sizeof(char) * 255, cudaMemcpyHostToDevice);
        cudaMalloc((void **) &material, sizeof(Material));
        cudaMemcpy(material, mesh->material, sizeof(Material), cudaMemcpyHostToDevice);
        cudaMalloc((void **) &material_texture_id, sizeof(int));
        cudaMemcpy(material_texture_id, &mesh->material_texture_id, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void **) &devicePointer, sizeof(DeviceTriangleMesh));
        cudaMemcpy(devicePointer, this, sizeof(DeviceTriangleMesh), cudaMemcpyHostToDevice);
    }

    void free() {
        delete vertex_device_vector;
        delete normal_device_vector;
        delete texcoord_device_vector;
        delete index_device_vector;
        cudaFree(name);
        cudaFree(material);
        cudaFree(material_texture_id);
    }

    DeviceTriangleMesh * devicePointer;
    /**
     * 그룹 이름.
     */
    char* name;

    /**
     * 그룹에 속한 정점 목록.
     */
    thrust::device_vector<glm::vec3>* vertex_device_vector;
    glm::vec3* vertex;
    /**
     * 그룹에 속한 정점의 법선 벡터.
     */
    thrust::device_vector<glm::vec3>* normal_device_vector;
    glm::vec3* normal;

    /**
     * 그룹에 속한 정점의 텍스쳐 좌표 정보.
     */
    thrust::device_vector<glm::vec2>* texcoord_device_vector;
    glm::vec2* texcoord;
    /**
     * 그룹에 속한 정점들로 이뤄지는 면에 대한 정보.
     */
    thrust::device_vector<glm::ivec3>* index_device_vector;
    glm::ivec3* index;

    /**
     * 그룹에 일괄적으로 적용되는 재질.
     * 오브젝트의 각 부분을 색상 등으로 구분하기 위해 사용.
     */
    Material* material;

    /**
     * 그룹에 적용되는 재질에 대응하는 인덱스.
     */
    int* material_texture_id;
};

#endif
