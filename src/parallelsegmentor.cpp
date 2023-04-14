#include "parallelsegmentor.h"

ParallelSegmentor::ParallelSegmentor(TriangleMesh* mesh, float tolerance)
    : Segmentor(mesh, tolerance) {
}

std::vector<TriangleMesh*> ParallelSegmentor::do_segmentation() {
    std::vector<TriangleMesh*> vector;
    return vector;
};
