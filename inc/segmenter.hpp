#ifndef __SEGMENTER_H
#define __SEGMENTER_H

#define TIMER_PREPROCESSING 0
#define TIMER_NORMAL_VECTOR_COMPUTATION 1
#define TIMER_MAP_COUNT 2
#define TIMER_NORMAL_MAP_INSERTION 3
#define TIMER_CC_N_TMG 4
#define TIMER_FACEGRAPH_INIT_A 5
#define TIMER_FACEGRAPH_INIT_B 6
#define TIMER_FACEGRAPH_GET_SETMENTS_A 7
#define TIMER_FACEGRAPH_GET_SETMENTS_B 8
#define TIMER_TRIANGLE_MESH_GENERATING 9
#define TIMER_SEGMENT_COLORING 10
#define TIMER_TOTAL 11
#define TIMER_DATA_TRANSFER_D2H 12
#define TIMER_SIZE (TIMER_DATA_TRANSFER_D2H + 1)

#include "dstimer.hpp"
#include "trianglemesh.hpp"
#include <vector>
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
    TriangleMesh* mesh;
    /**
     * 세그멘테이션의 기준이 되는 각도.
     * 메시 그룹에 속한 면에 대한 법선 벡터의 각도를 이용해 세그멘테이션을
     * 수행한다.
     */
    float tolerance;

  public:
    Segmenter(TriangleMesh* mesh, float tolerance = 0.0f)
        : mesh(mesh), tolerance(tolerance), timer(DS_timer(TIMER_SIZE)) {
        timer.setTimerName(TIMER_PREPROCESSING,
                           (char*)"Preprocessing                                     ");
        timer.setTimerName(TIMER_NORMAL_VECTOR_COMPUTATION,
                           (char*)"  - Normal Vector Computation                     ");
        timer.setTimerName(TIMER_MAP_COUNT,
                           (char*)"  - Map Count                                     ");
        timer.setTimerName(TIMER_NORMAL_MAP_INSERTION,
                           (char*)"  - Normal Map Insertion                          ");
        timer.setTimerName(TIMER_CC_N_TMG,
                           (char*)"Connectivity Checking and Triangle Mesh Generating");
        timer.setTimerName(TIMER_FACEGRAPH_INIT_A,
                           (char*)"  - FaceGraph: Init A                             ");
        timer.setTimerName(TIMER_FACEGRAPH_INIT_B,
                           (char*)"  - FaceGraph: Init B                             ");
        timer.setTimerName(TIMER_FACEGRAPH_GET_SETMENTS_A,
                           (char*)"  - FaceGraph: Get Segments A                     ");
        timer.setTimerName(TIMER_FACEGRAPH_GET_SETMENTS_B,
                           (char*)"  - FaceGraph: Get Segments B                     ");
        timer.setTimerName(TIMER_TRIANGLE_MESH_GENERATING,
                           (char*)"  - Triangle Mesh Generating                      ");
        timer.setTimerName(TIMER_SEGMENT_COLORING,
                           (char*)"Segment Coloring                                  ");
        timer.setTimerName(TIMER_TOTAL,
                           (char*)"Total (Preprocessing + CC & TMG)                  ");
        timer.setTimerName(TIMER_DATA_TRANSFER_D2H,
                           (char*)"Data Transfer Device To Host                      ");
    }
    virtual ~Segmenter(){};

    /**
     * @brief 세그멘테이션 수행.
     */
    virtual std::vector<TriangleMesh*> do_segmentation() = 0;
};

#endif
