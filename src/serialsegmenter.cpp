#include "serialsegmenter.h"

SerialSegmenter::SerialSegmenter(TriangleMesh* mesh, float tolerance)
    : Segmenter(mesh, tolerance) {
}

std::vector<TriangleMesh*> SerialSegmenter::do_segmentation() {
    std::vector<glm::vec3> face_normals(mesh->index.size());

    for (int i = 0; i < mesh->index.size(); i++) {
        face_normals[i] = glm::triangleNormal(mesh->vertex[mesh->index[i].x],
                                              mesh->vertex[mesh->index[i].y],
                                              mesh->vertex[mesh->index[i].z]);
    }

    std::cout << "Normal vector compute done" << std::endl;

    size_t count = face_normals.size();

    std::unordered_map<glm::vec3, size_t, FaceGraph::Vec3Hash> count_map;

    for (int i = 0; i < count; i++) {
        glm::vec3 target_norm = face_normals[i];

        for (auto iter : count_map) {
            glm::vec3 compare = iter.first;
            float norm_angle = glm::degrees(glm::angle(compare, target_norm));

            if (norm_angle < tolerance) {
                target_norm = compare;
                break;
            }
        }

        auto item = count_map.find(target_norm);

        if (item == count_map.end()) {
            count_map.insert({target_norm, 0});
        }
        count_map[target_norm]++;
    }

    std::unordered_map<glm::vec3, std::vector<FaceGraph::Triangle>,
                       FaceGraph::Vec3Hash>
        my_map;
    for (auto& iter : count_map) {
        // std::cout << "map["<< glm::to_string(iter.first) <<"] : " <<
        // iter.second << std::endl;
        my_map.insert(
            {iter.first, std::vector<FaceGraph::Triangle>(iter.second)});
        // std::cout << "check : " << my_map[iter.first].size() << std::endl;
        iter.second = 0;
    }
    std::cout << "Map count complete (map size : " << count_map.size() << ")"
              << std::endl;

    for (int i = 0; i < count; i++) {
        glm::vec3 target_norm = face_normals[i];

        for (auto& iter : count_map) {
            glm::vec3 compare = iter.first;
            float norm_angle = glm::degrees(glm::angle(compare, target_norm));

            if (norm_angle < tolerance) {
                target_norm = compare;
                break;
            }
        }

        auto indexes = count_map.find(target_norm);

        indexes->second++;
    }

    for (auto& iter : count_map) {
        if (my_map[iter.first].size() != iter.second) {
            // std::cout << "Validation steps: " << std::endl;
            // std::cout << "map[" << glm::to_string(iter.first) << "] : "
            // << iter.second << std::endl; std::cout << "check : " <<
            // my_map[iter.first].size() << std::endl;
            my_map[iter.first].resize(iter.second);
        }

        if (iter.second == 0) {
            std::cout << "Removed map : " << glm::to_string(iter.first) << " "
                      << iter.second << std::endl;
            my_map.erase(iter.first);
            count_map.erase(iter.first);
        }

        iter.second = 0;
    }
    std::cout << "Map compaction complete (map size : " << count_map.size()
              << ")" << std::endl;

    double total_time = 0.0;
    for (int i = 0; i < count; i++) {
        glm::vec3 target_norm = face_normals[i];

        for (auto& iter : count_map) {
            glm::vec3 compare = iter.first;
            float norm_angle = glm::degrees(glm::angle(compare, target_norm));

            if (norm_angle < tolerance) {
                target_norm = compare;
                break;
            }
        }

        auto item = my_map.find(target_norm);
        auto indexes = count_map.find(target_norm);

        FaceGraph::Triangle tri;
        tri.vertex[0] = mesh->vertex[mesh->index[i].x];
        tri.vertex[1] = mesh->vertex[mesh->index[i].y];
        tri.vertex[2] = mesh->vertex[mesh->index[i].z];

        item->second[indexes->second++] = tri;
    }

    std::cout << "Normal map insert done total (" << my_map.size()
              << ") size map" << std::endl;

    // std::cout << "map size : " << my_map.size() << std::endl;

    std::vector<TriangleMesh*> result;
    int number = 0;
    for (auto iter : my_map) {
        auto start_time = std::chrono::system_clock::now();
        // std::cout << "Key[" << iter.first.x << ", " << iter.first.y << ",
        // " << iter.first.z << "] : "; std::cout << "Number of faces : " <<
        // iter.second.size() << std::endl;

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

    my_map.clear();

    return result;
};
