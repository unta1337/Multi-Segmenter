#ifndef __PARALLELSEGMENTER_H
#define __PARALLELSEGMENTER_H

#include "segmenter.hpp"
#include "trianglemesh.hpp"
#include <vector>

class ParallelSegmenter : public Segmenter {
  public:
    ParallelSegmenter(TriangleMesh* mesh, float tolerance = 0.0f);
    virtual std::vector<TriangleMesh*> do_segmentation();
};

#endif
