#ifndef __PARALLELSEGMENTOR_H
#define __PARALLELSEGMENTOR_H

#include "segmentor.hpp"
#include "trianglemesh.hpp"
#include <vector>

class ParallelSegmentor : public Segmentor {
  public:
    ParallelSegmentor(TriangleMesh* mesh, float tolerance = 0.0f);
    virtual std::vector<TriangleMesh*> do_segmentation();
};

#endif
