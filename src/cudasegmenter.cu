#include "cudafacegraph.h"
#include "cudasegmenter.h"

CUDASegmenter::CUDASegmenter(TriangleMesh* mesh, float tolerance) : Segmenter(mesh, tolerance) {
    timer.onTimer(TIMER_DATA_TRANSFER_D2H);
    deviceMesh = new DeviceTriangleMesh(mesh);
    timer.offTimer(TIMER_DATA_TRANSFER_D2H);
}

CUDASegmenter::~CUDASegmenter() {
    deviceMesh->free();
    free(deviceMesh);
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
struct NormalWrapper {
    glm::vec3 normal;
    float xAngle;
    float yAngle;
    float zAngle;
    Triangle triangle;
};

struct NormalMapper {
    glm::vec3* vertex;
    glm::vec3 xAxis = glm::vec3(1, 0, 0);
    glm::vec3 yAxis = glm::vec3(1, 0, 0);
    glm::vec3 zAxis = glm::vec3(1, 0, 0);

    explicit NormalMapper(glm::vec3* vertex) : vertex(vertex) {
    }

    __host__ __device__ NormalWrapper operator()(const glm::ivec3& idx) const {
        Triangle triangle;
        triangle.vertex[0] = vertex[idx[0]];
        triangle.vertex[1] = vertex[idx[1]];
        triangle.vertex[2] = vertex[idx[2]];
        glm::vec3 normal = glm::triangleNormal(triangle.vertex[0], triangle.vertex[1], triangle.vertex[2]);
        float xAngle = glm::angle(normal, xAxis);
        float yAngle = glm::angle(normal, yAxis);
        float zAngle = glm::angle(normal, zAxis);
        return {normal, xAngle, yAngle, zAngle, triangle};
    }
};

struct NormalIndexMapper {
    float tolerance;
    int baseSize;

    explicit NormalIndexMapper(float tolerance) {
        baseSize = floor(180.0f / tolerance);
        this->tolerance = glm::radians(tolerance);
    }

    __host__ __device__ int operator()(const NormalWrapper& normal) const {
        return ((int)floor(normal.xAngle / tolerance)) + ((int)floor(normal.yAngle / tolerance)) * baseSize +
               ((int)floor(normal.zAngle / tolerance)) * baseSize * baseSize;
    }
};

struct NormalTriangleMapper {
    explicit NormalTriangleMapper() {
    }

    __host__ __device__ Triangle operator()(const NormalWrapper& normal) const {
        return normal.triangle;
    }
};

struct AngleComparator {
    __host__ __device__ bool operator()(const NormalWrapper& o1, const NormalWrapper& o2) const {
        if (o1.xAngle < o2.xAngle)
            return true;
        else if (o1.xAngle > o2.xAngle)
            return false;

        // If xAngle is equal, then compare yAngle
        if (o1.yAngle < o2.yAngle)
            return true;
        else if (o1.yAngle > o2.yAngle)
            return false;

        // If yAngle is also equal, then compare zAngle
        if (o1.zAngle < o2.zAngle)
            return true;
        else
            return false;
    }
};

std::vector<TriangleMesh*> CUDASegmenter::do_segmentation() {
    timer.onTimer(TIMER_TOTAL);
    STEP_LOG(std::cout << "[Begin] Preprocessing.\n");
    timer.onTimer(TIMER_PREPROCESSING);
    STEP_LOG(std::cout << "[Begin] Normal Vector Computation.\n");
    timer.onTimer(TIMER_NORMAL_VECTOR_COMPUTATION);

    // obj에 포함된 면의 개수만큼 법선 벡터 계산 필요.
    thrust::device_vector<NormalWrapper> face_normals(mesh->index.size());
    thrust::transform(deviceMesh->index_device_vector->begin(), deviceMesh->index_device_vector->end(),
                      face_normals.begin(), NormalMapper(deviceMesh->vertex));

    timer.offTimer(TIMER_NORMAL_VECTOR_COMPUTATION);

    STEP_LOG(std::cout << "[End] Normal Vector Computation.\n");

    thrust::sort(face_normals.begin(), face_normals.end(), AngleComparator());
    thrust::device_vector<int> fn_indexes(face_normals.size());
    thrust::transform(face_normals.begin(), face_normals.end(), fn_indexes.begin(), NormalIndexMapper(tolerance));

    int baseSize = ceil(180.0f / tolerance);
    int binSize = baseSize * baseSize * baseSize;
    thrust::device_vector<int> indexes(binSize);
    thrust::device_vector<int> counts(binSize);
    thrust::reduce_by_key(fn_indexes.begin(), fn_indexes.end(), thrust::make_constant_iterator(1), indexes.begin(),
                          counts.begin(), thrust::equal_to<int>(), thrust::plus<int>());

    thrust::device_vector<Triangle> fn_triangles(face_normals.size());
    thrust::transform(face_normals.begin(), face_normals.end(), fn_triangles.begin(), NormalTriangleMapper());
    timer.offTimer(TIMER_PREPROCESSING);
    STEP_LOG(std::cout << "[End] Preprocessing.\n");

    STEP_LOG(std::cout << "[Begin] Connectivity Checking and Triangle Mesh Generating.\n");
    timer.onTimer(TIMER_CC_N_TMG);

    std::vector<int> startIndexes(binSize);
    int startIndex = 0;
    for (int i = 1; i < indexes.size(); i++) {
        startIndexes[i] = (startIndex += counts[i - 1]);
    }

    std::vector<TriangleMesh*> result;
    // 이제 병렬화가 가능할 것으로 보임
    int number = 0;
    for (int i = 0; i < binSize; i++) {
        int start = startIndexes[i];
        int end = start + counts[i];
        STEP_LOG(std::cout << "[Step] FaceGraph: Init.\n");
        std::vector<Triangle> triangles(counts[i]);
        thrust::copy(fn_triangles.begin() + start, fn_triangles.begin() + end, triangles.begin());
        // 오버헤드 제거 가능

        CUDAFaceGraph fg(&triangles, &timer);

        STEP_LOG(std::cout << "[Step] FaceGraph: Get Segments.\n");
        std::vector<std::vector<Triangle>> segments = fg.get_segments();

        STEP_LOG(std::cout << "[Step] Triangle Mesh Generating.\n");
        timer.onTimer(TIMER_TRIANGLE_MESH_GENERATING);
        for (const auto& segment : segments) {
            TriangleMesh* sub_object = triangle_list_to_obj(segment);
            sub_object->material->diffuse = glm::vec3(1, 0, 0);
            strcpy(sub_object->material->name, ("sub_materials_" + std::to_string(number)).c_str());
            strcpy(sub_object->name, (std::string(mesh->name) + "_seg_" + std::to_string(number++)).c_str());
            result.push_back(sub_object);
        }
        timer.offTimer(TIMER_TRIANGLE_MESH_GENERATING);
    }

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

    //    normal_triangle_list_map.clear();

    timer.offTimer(TIMER_TOTAL);
    return result;
};
