#include "parallelsegmenter.h"

ParallelSegmenter::ParallelSegmenter(TriangleMesh* mesh, float tolerance)
    : Segmenter(mesh, tolerance) {
}

std::vector<TriangleMesh*> ParallelSegmenter::do_segmentation() {
    std::vector<TriangleMesh*> vector;
    return vector;
};
