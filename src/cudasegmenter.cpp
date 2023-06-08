#include "cudasegmenter.h"
#include "cudafacegraph.h"

CUDASegmenter::CUDASegmenter(TriangleMesh* mesh, float tolerance) : Segmenter(mesh, tolerance) {
}

inline glm::vec3 CUDASegmenter::get_normal_key(std::unordered_map<glm::vec3, size_t, Vec3Hash>& count_map,
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

inline void CUDASegmenter::init_count_map(std::unordered_map<glm::vec3, size_t, Vec3Hash>& count_map,
                                          std::vector<glm::vec3>& face_normals) {
    for (auto& normal : face_normals) {
        count_map[get_normal_key(count_map, normal)]++;
    }
}

std::vector<TriangleMesh*> CUDASegmenter::do_segmentation() {
    timer.onTimer(TIMER_TOTAL);
    STEP_LOG(std::cout << "[Begin] Preprocessing.\n");
    timer.onTimer(TIMER_PREPROCESSING);
    STEP_LOG(std::cout << "[Begin] Normal Vector Computation.\n");
    timer.onTimer(TIMER_NORMAL_VECTOR_COMPUTATION);

    // obj에 포함된 면의 개수만큼 법선 벡터 계산 필요.
    std::vector<glm::vec3> face_normals(mesh->index.size());

    // 오브젝트에 포함된 면에 대한 법선 벡터 계산.
    for (int i = 0; i < mesh->index.size(); i++) {
        glm::ivec3& index = mesh->index[i];
        face_normals[i] = glm::triangleNormal(mesh->vertex[index[0]], mesh->vertex[index[1]], mesh->vertex[index[2]]);
    }

    timer.offTimer(TIMER_NORMAL_VECTOR_COMPUTATION);
    STEP_LOG(std::cout << "[End] Normal Vector Computation.\n");

    STEP_LOG(std::cout << "[Begin] Map Count.\n");
    timer.onTimer(TIMER_MAP_COUNT);

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

    timer.offTimer(TIMER_MAP_COUNT);
    STEP_LOG(std::cout << "[End] Map Count. (Map size: " << count_map.size() << ")\n");

    STEP_LOG(std::cout << "[Begin] Normal Map Insertion.\n");
    timer.onTimer(TIMER_NORMAL_MAP_INSERTION);

    for (int i = 0; i < face_normals_count; i++) {
        glm::vec3 target_normal = get_normal_key(count_map, face_normals[i]);

        Triangle triangle;
        glm::ivec3 indexes = mesh->index[i];
        for (int d = 0; d < 3; d++) {
            triangle.vertex[d] = mesh->vertex[indexes[d]];
            triangle.id[d] = indexes[d];
        }

        normal_triangle_list_map[target_normal].push_back(triangle);
    }

    timer.offTimer(TIMER_NORMAL_MAP_INSERTION);
    STEP_LOG(std::cout << "[End] Normal Map Insertion. (Total: " << normal_triangle_list_map.size() << ")\n");

    timer.offTimer(TIMER_PREPROCESSING);
    STEP_LOG(std::cout << "[End] Preprocessing.\n");

    STEP_LOG(std::cout << "[Begin] Connectivity Checking and Triangle Mesh Generating.\n");
    timer.onTimer(TIMER_CC_N_TMG);

    std::vector<std::pair<std::vector<int>, std::vector<Triangle>*>> segments_collection;
    for (auto& entry : normal_triangle_list_map) {
        STEP_LOG(std::cout << "[Step] FaceGraph: Init.\n");
        CUDAFaceGraph fg(&entry.second, &timer);

        STEP_LOG(std::cout << "[Step] FaceGraph: Get Segments.\n");
        segments_collection.push_back(std::make_pair(fg.get_segments_as_union(), fg.triangles));
    }

    STEP_LOG(std::cout << "[Step] Triangle Mesh Generating.\n");
    timer.onTimer(TIMER_TRIANGLE_MESH_GENERATING);

    std::vector<TriangleMesh*> result;
    for (int i = 0; i < segments_collection.size(); i++) {
        auto& [segments, triangles] = segments_collection[i];
        std::vector<TriangleMesh*> sub_obj = segment_union_to_obj(segments, triangles, mesh->vertex.size());
        result.insert(result.end(), sub_obj.begin(), sub_obj.end());
    }

    for (int i = 0; i < result.size(); i++) {
        TriangleMesh* sub_object = result[i];
        sub_object->material->diffuse = glm::vec3(1, 0, 0);
        sub_object->material->name = "sub_materials_" + std::to_string(i);
        sub_object->name = mesh->name + "_seg_" + std::to_string(i);
    }

    timer.offTimer(TIMER_TRIANGLE_MESH_GENERATING);

    timer.offTimer(TIMER_CC_N_TMG);
    STEP_LOG(std::cout << "[End] Connectivity Checking and Triangle Mesh Generating.\n");

    STEP_LOG(std::cout << "[Begin] Segment Coloring.\n");
    timer.onTimer(TIMER_SEGMENT_COLORING);

    for (int i = 0; i < result.size(); i++) {
        result[i]->material->diffuse = Color::get_color_from_jet((float)i, 0, (float)result.size());
        result[i]->material->ambient = glm::vec3(1.0f, 1.0f, 1.0f);
        result[i]->material->specular = glm::vec3(0.5f, 0.5f, 0.5f);
    }

    STEP_LOG(std::cout << "[End] Segment Coloring.\n");
    timer.offTimer(TIMER_SEGMENT_COLORING);

    normal_triangle_list_map.clear();

    timer.offTimer(TIMER_TOTAL);
    return result;
};
