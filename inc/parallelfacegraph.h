#ifndef __PARALLELFACEGRAPH_H
#define __PARALLELFACEGRAPH_H

#include "facegraph.hpp"
#include <omp.h>

class ParallelFaceGraph : public FaceGraph {
  public:
    ParallelFaceGraph(std::vector<Triangle>* list);
    virtual std::vector<std::vector<Triangle>> get_segments();
    virtual void traverse_dfs(std::vector<int>& visit, int start_vert, int count);
};

#endif
