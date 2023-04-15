#ifndef __PARALLELFACEGRAPH_H
#define __PARALLELFACEGRAPH_H

#include "facegraph.hpp"
class ParallelFaceGraph : public FaceGraph {
  public:
    ParallelFaceGraph(std::vector<Triangle>* list);
    virtual std::vector<std::vector<Triangle>> check_connected();
    virtual void traverse_dfs(std::vector<int>& visit, int start_vert, int count);
};

#endif
