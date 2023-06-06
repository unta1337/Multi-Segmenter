#ifndef __SEGMENTER_H
#define __SEGMENTER_H

#include "dstimer.hpp"
#include "object.h"
#include "logutils.h"
#include "timer.h"
#include "color.hpp"
#include "triangle.h"
#include "trianglemesh.hpp"
#include "facegraph.hpp"
#include <vector>
#include <unordered_map>

/**
 * @brief 하나의 메시 그룹에 대한 세그멘테이션을 수행.
 */
class Segmenter {
public:
    DS_timer timer;

protected:
    /**
     * 세그멘테이션을 수행할 대상 메시 그룹.
     */
    Object *object;
    /**
     * 세그멘테이션의 기준이 되는 각도.
     * 메시 그룹에 속한 면에 대한 법선 벡터의 각도를 이용해 세그멘테이션을
     * 수행한다.
     */
    float tolerance;

public:
    Segmenter(Object *object, float tolerance = 0.0f) : object(object), tolerance(tolerance) {

    }

    virtual ~Segmenter() {};

    /**
     * @brief 세그멘테이션 수행.
     */
    template<typename UNKNOWN_TYPE = int>
    std::vector<UNKNOWN_TYPE> do_segmentation() {
        timer.onTimer(TIMER_TOTAL);
        STEP_LOG(std::cout << "[Begin] Preprocessing.\n");
        timer.onTimer(TIMER_PREPROCESSING);
        STEP_LOG(std::cout << "[Begin] Normal Vector Computation.\n");
        timer.onTimer(TIMER_NORMAL_VECTOR_COMPUTATION);

        // obj에 포함된 면의 개수만큼 법선 벡터 계산.
        calc_face_normals(*object);

        timer.offTimer(TIMER_NORMAL_VECTOR_COMPUTATION);
        STEP_LOG(std::cout << "[End] Normal Vector Computation.\n");

        STEP_LOG(std::cout << "[Begin] Map Count.\n");
        timer.onTimer(TIMER_MAP_COUNT);

        size_t face_normals_count = object->faces.size();

        // TODO: GPU로 변환할 것
        // 법선 벡터 -> 개수.
        // 특정 법선 벡터와 비슷한 방향성을 갖는 법선 벡터의 개수.

        // TODO: 현재 법선 별 구분을 하지 않고 모두 하나의 그룹으로 취급함. 관련 로직 구현 필요.
        std::unordered_map<Vertex, size_t> count_map;
        for (Face& face : object->faces) {
            count_map[{ face.nx, face.ny, face.nz }]++;
        }

        // 법선 벡터 -> 삼각형(면).
        // 특정 법선 벡터와 비슷한 방향성을 갖는 벡터를 법선 벡터로 갖는 면.
        std::unordered_map<Vertex, std::vector<Triangle>> normal_triangle_list_map;
        for (const auto& entry : count_map) {
            normal_triangle_list_map.insert({entry.first, std::vector<Triangle>()});
        }

        timer.offTimer(TIMER_MAP_COUNT);
        STEP_LOG(std::cout << "[End] Map Count. (Map size: " << count_map.size() << ")\n");

        STEP_LOG(std::cout << "[Begin] Normal Map Insertion.\n");
        timer.onTimer(TIMER_NORMAL_MAP_INSERTION);

        for (int i = 0; i < face_normals_count; i++) {
            // Vector3f target_normal = get_normal_key(count_map, face_normals[i]);
            // TODO: 기존 get_normal_key에 대응하는 로직 구현 필요.
            Vector3f target_normal = { object->faces[0].nx, object->faces[0].ny, object->faces[0].nz };

            Triangle triangle;
            Vector3u indexes{ object->faces[0].pi, object->faces[0].qi, object->faces[0].ri };
            triangle.vertex[0] = object->vertices[indexes.x];
            triangle.vertex[1] = object->vertices[indexes.y];
            triangle.vertex[2] = object->vertices[indexes.z];

            normal_triangle_list_map[target_normal].push_back(triangle);
        }

        timer.offTimer(TIMER_NORMAL_MAP_INSERTION);
        STEP_LOG(std::cout << "[End] Normal Map Insertion. (Total: " << normal_triangle_list_map.size() << ")\n");

        timer.offTimer(TIMER_PREPROCESSING);
        STEP_LOG(std::cout << "[End] Preprocessing.\n");

        STEP_LOG(std::cout << "[Begin] Connectivity Checking and Triangle Mesh Generating.\n");
        timer.onTimer(TIMER_CC_N_TMG);

        std::vector<TriangleMesh*> result;
        int number = 0;
        for (auto& entry : normal_triangle_list_map) {
            STEP_LOG(std::cout << "[Step] FaceGraph: Init.\n");
            FaceGraph fg(&entry.second, &timer);

            STEP_LOG(std::cout << "[Step] FaceGraph: Get Segments.\n");
            std::vector<std::vector<Triangle>> segments = fg.get_segments();

            STEP_LOG(std::cout << "[Step] Triangle Mesh Generating.\n");
            timer.onTimer(TIMER_TRIANGLE_MESH_GENERATING);
            for (const auto& segment : segments) {
                TriangleMesh* sub_object = triangle_list_to_obj(segment);
                sub_object->material->diffuse = {1, 0, 0};
                sub_object->material->name = "sub_materials_" + std::to_string(number);
                sub_object->name = mesh->name + "_seg_" + std::to_string(number++);

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
            result[i]->material->ambient = Vector3f(1.0f, 1.0f, 1.0f);
            result[i]->material->specular = Vector3f(0.5f, 0.5f, 0.5f);
        }

        STEP_LOG(std::cout << "[End] Segment Coloring.\n");
        timer.offTimer(TIMER_SEGMENT_COLORING);

        normal_triangle_list_map.clear();

        timer.offTimer(TIMER_TOTAL);
        return result;
    };
};

#endif
