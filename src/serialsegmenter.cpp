#include "serialsegmenter.h"

SerialSegmenter::SerialSegmenter(TriangleMesh* mesh, float tolerance)
    : Segmenter(mesh, tolerance) {
}

glm::vec3 SerialSegmenter::get_normal_key(
    std::unordered_map<glm::vec3, size_t, FaceGraph::Vec3Hash>& count_map,
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

void SerialSegmenter::init_count_map(
    std::unordered_map<glm::vec3, size_t, FaceGraph::Vec3Hash>& count_map,
    std::vector<glm::vec3>& face_normals) {
    for (auto& normal : face_normals) {
        count_map[get_normal_key(count_map, normal)]++;
    }
}

std::vector<TriangleMesh*> SerialSegmenter::do_segmentation() {
    std::vector<glm::vec3> face_normals(mesh->index.size());

    for (int i = 0; i < mesh->index.size(); i++) {
        glm::ivec3& index = mesh->index[i];
        face_normals[i] =
            glm::triangleNormal(mesh->vertex[index[0]], mesh->vertex[index[1]],
                                mesh->vertex[index[2]]);
    }

    std::cout << "Normal vector compute done" << std::endl;

    size_t face_normals_count = face_normals.size();

    std::unordered_map<glm::vec3, size_t, FaceGraph::Vec3Hash> count_map;
    init_count_map(count_map, face_normals);

    std::unordered_map<glm::vec3, std::vector<FaceGraph::Triangle>,
                       FaceGraph::Vec3Hash>
        normal_triangle_list_map;
    for (const auto& entry : count_map) {
        normal_triangle_list_map.insert(
            {entry.first, std::vector<FaceGraph::Triangle>()});
    }
    std::cout << "Map count complete (map size : " << count_map.size() << ")"
              << std::endl;

    double total_time = 0.0;
    for (int i = 0; i < face_normals_count; i++) {
        glm::vec3 target_norm = get_normal_key(count_map, face_normals[i]);

        FaceGraph::Triangle triangle;
        glm::ivec3 indexes = mesh->index[i];
        for (int d = 0; d < 3; d++) {
            triangle.vertex[d] = mesh->vertex[indexes[d]];
        }

        normal_triangle_list_map[target_norm].push_back(triangle);
    }

    std::cout << "Normal map insert done total (" << normal_triangle_list_map.size()
              << ") size map" << std::endl;

    std::vector<TriangleMesh*> result;
    int number = 0;
    for (auto iter : normal_triangle_list_map) {
        auto start_time = std::chrono::system_clock::now();

        FaceGraph::FaceGraph fg(&iter.second);
        std::cout << "Face Graph done" << std::endl;
        std::vector<std::vector<FaceGraph::Triangle>> temp =
            fg.check_connected();
        std::cout << "Check connected done" << std::endl;

        for (auto subs : temp) {
            TriangleMesh* sub_object = FaceGraph::triangle_list_to_obj(subs);
            sub_object->material->diffuse = glm::vec3(1, 0, 0);
            sub_object->material->name =
                "sub_materials_" + std::to_string(number);
            sub_object->name = mesh->name + "_seg_" + std::to_string(number++);

            result.push_back(sub_object);
        }

        auto end_time = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();

        total_time += (ms / 1000.);
        std::cout << "Spend : " << total_time << " ms (" << (ms / 1000.)
                  << " ms)" << std::endl;
    }
    std::cout << "Check connectivity and Make triangle mesh done" << std::endl;

    for (int i = 0; i < result.size(); i++) {
        result[i]->material->diffuse =
            Color::get_color_from_jet((float)i, 0, (float)result.size());
        result[i]->material->ambient = glm::vec3(1.0f, 1.0f, 1.0f);
        result[i]->material->specular = glm::vec3(0.5f, 0.5f, 0.5f);
    }

    normal_triangle_list_map.clear();

    return result;
};
