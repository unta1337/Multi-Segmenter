#ifndef __OBJECT_H
#define __OBJECT_H

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "face.h"
#include "vectex.h"

struct Object {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
};

struct DeviceObject {
    thrust::device_vector<Vertex> vertices;
    thrust::device_vector<Face> faces;
};
#endif