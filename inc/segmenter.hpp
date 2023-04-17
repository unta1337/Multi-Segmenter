#ifndef __SEGMENTER_H
#define __SEGMENTER_H

#include "trianglemesh.hpp"
#include "dstimer.h"
#include <vector>

/**
 * @brief 하나의 메시 그룹에 대한 세그멘테이션을 수행.
 */
class Segmenter {
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

    DS_timer timer;

  public:
    Segmenter(TriangleMesh* mesh, float tolerance = 0.0f) : mesh(mesh), tolerance(tolerance), timer(DS_timer(6)) {
      timer.setTimerName(0, (char*)"Normal Vector Computation                         ");
      timer.setTimerName(1, (char*)"Map Count                                         ");
      timer.setTimerName(2, (char*)"Normal Map Insertion                              ");
      timer.setTimerName(3, (char*)"Connectivity Checking and Triangle Mesh Generating");
      timer.setTimerName(4, (char*)"temp1                                             ");
      timer.setTimerName(5, (char*)"temp2                                             ");
    }
    virtual ~Segmenter(){};

    virtual DS_timer get_timer() {
      return timer;
    }

    /**
     * @brief 세그멘테이션 수행.
     */
    virtual std::vector<TriangleMesh*> do_segmentation() = 0;
};

#endif
