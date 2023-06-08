#ifndef __CUDAFACEGRAPH_H
#define __CUDAFACEGRAPH_H

#include "facegraph.hpp"
#include "segmenter.hpp"

class CUDAFaceGraph : public FaceGraph {
  public:
    /**
     * 삼각형에 대한 인접 리스트
     */
    std::vector<std::vector<int>> adj_triangles;
    CUDAFaceGraph(std::vector<Triangle>* triangles);
    CUDAFaceGraph(std::vector<Triangle>* triangles, DS_timer* timer);
    CUDAFaceGraph(std::vector<Triangle>* triangles, DS_timer* timer, std::vector<std::vector<int>>& vertex_to_adj);
    virtual void init();
    virtual void init(std::vector<std::vector<int>>& vertex_to_id);
    virtual std::vector<std::vector<Triangle>> get_segments();
    virtual std::vector<int> get_segments_as_union();
    virtual void traverse_dfs(std::vector<int>& visit, int start_vert, int count);
};

#endif
