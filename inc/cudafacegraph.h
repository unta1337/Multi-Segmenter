#ifndef __CUDAFACEGRAPH_H
#define __CUDAFACEGRAPH_H

#include "facegraph.hpp"
#include "segmenter.hpp"

#include <thrust/device_vector.h>

class CUDAFaceGraph : public FaceGraph {
  public:
    /**
     * 삼각형에 대한 인접 리스트
     */
    std::vector<std::vector<int>> adj_triangles;
    size_t* d_triangles_soa;
    int total_vertex_count;
    CUDAFaceGraph(std::vector<Triangle>* triangles);
    CUDAFaceGraph(std::vector<Triangle>* triangles, DS_timer* timer);
    CUDAFaceGraph(std::vector<Triangle>* triangles, DS_timer* timer, int total_vertex_count);
    virtual void init();
    virtual std::vector<std::vector<Triangle>> get_segments();
    virtual std::vector<int> get_segments_as_union();
    virtual void traverse_dfs(std::vector<int>& visit, int start_vert, int count);
    virtual std::vector<std::vector<int>> get_vertex_to_adj();
};

#endif
