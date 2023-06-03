#ifndef __DTYPES_H
#define __DTYPES_H

#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct vertex_t {
    float x;
    float y;
    float z;
};

struct face_t {
    size_t pi;
    size_t qi;
    size_t ri;
    float nx;
    float ny;
    float nz;
    float diff;
};

struct object_t {
    std::vector<vertex_t> vertices;
    std::vector<face_t> faces;
};

struct object_dt {
    thrust::device_vector<vertex_t> vertices;
    thrust::device_vector<face_t> faces;
};

#endif
