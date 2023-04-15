#include "parallelsegmenter.h"
#include <chrono>
#include <color.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/normal.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <iostream>
#include <omp.h>
#include <unordered_set>

#include <dstimer.h>

ParallelSegmenter::ParallelSegmenter(TriangleMesh* mesh, float tolerance) : Segmenter(mesh, tolerance) {
}

inline glm::vec3 ParallelSegmenter::get_normal_key(std::unordered_map<glm::vec3, size_t, Vec3Hash>& count_map,
                                                   glm::vec3& normal) {
    for (const auto& entry : count_map) {
        glm::vec3 compare = entry.first;
        float norm_angle = glm::degrees(glm::angle(compare, normal));

        if (norm_angle < tolerance) {
            normal = compare;
            break;
        }
    }
    return normal;
}
inline void ParallelSegmenter::init_count_map(std::unordered_map<glm::vec3, size_t, Vec3Hash>& count_map,
                                              std::vector<glm::vec3>& face_normals) {
    for (auto& normal : face_normals) {
        count_map[get_normal_key(count_map, normal)]++;
    }
}

std::vector<TriangleMesh*> ParallelSegmenter::do_segmentation() {
    DS_timer timer(1);
    timer.setTimerName(0, (char*)"Parallel Timer");
    timer.onTimer(0);
    omp_set_num_threads(5);

    // obj에 포함된 면의 개수만큼 법선 벡터 계산 필요.
    std::vector<glm::vec3> face_normals(mesh->index.size());

    // 오브젝트에 포함된 면에 대한 법선 벡터 계산.
#pragma omp parallel for
    for (int i = 0; i < mesh->index.size(); i++) {
        glm::ivec3& index = mesh->index[i];
        face_normals[i] = glm::triangleNormal(mesh->vertex[index[0]], mesh->vertex[index[1]], mesh->vertex[index[2]]);
    }

    std::cout << "Normal vector compute done" << std::endl;

    size_t face_normals_count = face_normals.size();

    // 법선 벡터 -> 개수.
    // 특정 법선 벡터와 비슷한 방향성을 갖는 법선 벡터의 개수.
    std::unordered_map<glm::vec3, size_t, Vec3Hash> count_map;
    // -- 초기 6개 잡을 때까지 countMap에 정보가 없어서 7~8개씩 잡히는 이슈.
    init_count_map(count_map, face_normals);

    // 법선 벡터 -> 삼각형(면).
    // 특정 법선 벡터와 비슷한 방향성을 갖는 벡터를 법선 벡터로 갖는 면.
    std::unordered_map<glm::vec3, std::vector<Triangle>, Vec3Hash> normal_triangle_list_map;
    std::unordered_set<glm::vec3, Vec3Hash> normal_list;
    // -- for iterator 지원을 안하고, 6개 밖게 안되서 하는 이유가 없을 것 같음. , insert라 serial로 하는게 속도가 더빠를
    // 듯.
    for (const auto& entry : count_map) {
        normal_triangle_list_map.insert({entry.first, std::vector<Triangle>()});
        normal_list.insert(entry.first);
    }
    std::cout << "Map count complete (map size : " << count_map.size() << ")" << std::endl;

    // 병렬을 위한 상호배제
    std::unordered_map<glm::vec3, omp_lock_t, Vec3Hash> lock_list;
    for (auto& it : normal_list) {
        lock_list.emplace(it, omp_lock_t());
        omp_init_lock(&lock_list[it]);
    }

#pragma omp parallel for
    for (int i = 0; i < face_normals_count; i++) {
        glm::vec3 target_norm = get_normal_key(count_map, face_normals[i]);

        Triangle triangle;
        glm::ivec3 indexes = mesh->index[i];
        for (int d = 0; d < 3; d++) {
            triangle.vertex[d] = mesh->vertex[indexes[d]];
        }

        omp_set_lock(&lock_list[target_norm]);
        normal_triangle_list_map[target_norm].push_back(triangle);
        omp_unset_lock(&lock_list[target_norm]);
    }

    std::cout << "Normal map insert done total (" << normal_triangle_list_map.size() << ") size map" << std::endl;

    std::vector<TriangleMesh*> result;
    int number = 0;
    for (auto iter : normal_triangle_list_map) {

        SerialFaceGraph fg(&iter.second);
        std::cout << "Face Graph done" << std::endl;
        std::vector<std::vector<Triangle>> temp = fg.get_segments();
        std::cout << "Check connected done" << std::endl;

        for (auto subs : temp) {
            TriangleMesh* sub_object = triangle_list_to_obj(subs);
            sub_object->material->diffuse = glm::vec3(1, 0, 0);
            sub_object->material->name = "sub_materials_" + std::to_string(number);
            sub_object->name = mesh->name + "_seg_" + std::to_string(number++);

            result.push_back(sub_object);
        }
    }
    std::cout << "Check connectivity and Make triangle mesh done" << std::endl;

#pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        result[i]->material->diffuse = Color::get_color_from_jet((float)i, 0, (float)result.size());
        result[i]->material->ambient = glm::vec3(1.0f, 1.0f, 1.0f);
        result[i]->material->specular = glm::vec3(0.5f, 0.5f, 0.5f);
    }

    for (auto& it : normal_list) {
        lock_list.emplace(it, omp_lock_t());
        omp_destroy_lock(&lock_list[it]);
    }

    normal_triangle_list_map.clear();
    timer.offTimer(0);
    timer.printTimer();
    return result;
};
