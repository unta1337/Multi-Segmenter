#include "parallelsegmenter.h"
#include "parallelfacegraph.h"

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
    STEP_LOG(std::cout << "[Begin] Normal Vector Computation.\n");
    timer.onTimer(0);

    // obj에 포함된 면의 개수만큼 법선 벡터 계산 필요.
    std::vector<glm::vec3> face_normals(mesh->index.size());

    // 오브젝트에 포함된 면에 대한 법선 벡터 계산.
    for (int i = 0; i < mesh->index.size(); i++) {
        glm::ivec3& index = mesh->index[i];
        face_normals[i] = glm::triangleNormal(mesh->vertex[index[0]], mesh->vertex[index[1]], mesh->vertex[index[2]]);
    }

    timer.offTimer(0);
    STEP_LOG(std::cout << "[End] Normal Vector Computation.\n");

    STEP_LOG(std::cout << "[Begin] Map Count.\n");
    timer.onTimer(1);

    size_t face_normals_count = face_normals.size();

    // 법선 벡터 -> 개수.
    // 특정 법선 벡터와 비슷한 방향성을 갖는 법선 벡터의 개수.
    std::unordered_map<glm::vec3, size_t, Vec3Hash> count_map;
    init_count_map(count_map, face_normals);

    // 법선 벡터 -> 삼각형(면).
    // 특정 법선 벡터와 비슷한 방향성을 갖는 벡터를 법선 벡터로 갖는 면.
    std::unordered_map<glm::vec3, std::vector<Triangle>, Vec3Hash> normal_triangle_list_map;
    for (const auto& entry : count_map) {
        normal_triangle_list_map.insert({entry.first, std::vector<Triangle>()});
    }

    timer.offTimer(1);
    STEP_LOG(std::cout << "[End] Map Count. (Map size: " << count_map.size() << ")\n");

    STEP_LOG(std::cout << "[Begin] Normal Map Insertion.\n");
    timer.onTimer(2);

    double total_time = 0.0;
    for (int i = 0; i < face_normals_count; i++) {
        glm::vec3 target_norm = get_normal_key(count_map, face_normals[i]);

        Triangle triangle;
        glm::ivec3 indexes = mesh->index[i];
        for (int d = 0; d < 3; d++) {
            triangle.vertex[d] = mesh->vertex[indexes[d]];
        }

        normal_triangle_list_map[target_norm].push_back(triangle);
    }

    timer.offTimer(2);
    STEP_LOG(std::cout << "[End] Normal Map Insertion. (Total: " << normal_triangle_list_map.size() << ")\n");

    STEP_LOG(std::cout << "[Begin] Connectivity Checking and Triangle Mesh Generating.\n");
    timer.onTimer(3);

    std::vector<TriangleMesh*> result;
    int number = 0;
    for (auto iter : normal_triangle_list_map) {
        ParallelFaceGraph fg(&iter.second, &timer);
        STEP_LOG(std::cout << "[Step] Face Graph Generating.\n");

        std::vector<std::vector<Triangle>> temp = fg.get_segments();
        STEP_LOG(std::cout << "[Step] Connectivity Checking.\n");

        for (auto subs : temp) {
            TriangleMesh* sub_object = triangle_list_to_obj(subs);
            sub_object->material->diffuse = glm::vec3(1, 0, 0);
            sub_object->material->name = "sub_materials_" + std::to_string(number);
            sub_object->name = mesh->name + "_seg_" + std::to_string(number++);

            result.push_back(sub_object);
        }
    }

    timer.offTimer(3);
    STEP_LOG(std::cout << "[End] Connectivity Checking and Triangle Mesh Generating.\n");

    for (int i = 0; i < result.size(); i++) {
        result[i]->material->diffuse = Color::get_color_from_jet((float)i, 0, (float)result.size());
        result[i]->material->ambient = glm::vec3(1.0f, 1.0f, 1.0f);
        result[i]->material->specular = glm::vec3(0.5f, 0.5f, 0.5f);
    }

    normal_triangle_list_map.clear();

    return result;
};
