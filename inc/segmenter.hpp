#ifndef __SEGMENTER_H
#define __SEGMENTER_H

#include "trianglemesh.hpp"
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

  public:
    Segmenter(TriangleMesh* mesh, float tolerance = 0.0f) : mesh(mesh), tolerance(tolerance) {
    }
    virtual ~Segmenter(){};

    /**
     * @brief 세그멘테이션 수행.
     */
    virtual std::vector<TriangleMesh*> do_segmentation() = 0;
};

#endif
