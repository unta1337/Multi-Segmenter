#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <unordered_map>
#include <vector>
#include "triangle.h"
#include <dstimer.hpp>

class TriangleMesh;

std::unordered_map<unsigned int, std::vector<Triangle>> kernelCall(TriangleMesh* mesh, float tolerance, DS_timer& timer);