﻿#include "parallelfacegraph.h"

ParallelFaceGraph::ParallelFaceGraph(std::vector<Triangle>* list) : FaceGraph(list) {
}

std::vector<std::vector<Triangle>> ParallelFaceGraph::get_segments() {
    std::vector<std::vector<Triangle>> vector;
    return vector;
}

void ParallelFaceGraph::traverse_dfs(std::vector<int>& visit, int start_vert, int count) {
}