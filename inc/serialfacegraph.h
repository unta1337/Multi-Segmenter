#ifndef __SERIALFACEGRAPH_H
#define __SERIALFACEGRAPH_H

#include "facegraph.hpp"

class SerialFaceGraph : public FaceGraph {
  public:
    SerialFaceGraph(std::vector<Triangle>* triangles);
    SerialFaceGraph(std::vector<Triangle>* triangles, DS_timer* timer);
    virtual void init();
    virtual std::vector<std::vector<Triangle>> get_segments();
    virtual void traverse_dfs(std::vector<int>& visit, int start_vert, int count);
};

#endif
