#include "objutils.h"

__host__ __device__ void __calc_normal(Face& face, Vector3f* vertices) {
    // 대응하는 정점 설정.
    Vector3f* v1 = &vertices[face.pi];
    Vector3f* v2 = &vertices[face.qi];
    Vector3f* v3 = &vertices[face.ri];

    // v2를 중심으로 한 방향 벡터 계산.
    Vector3f l1 = {v1->x - v2->x, v1->y - v2->y, v1->z - v2->z };
    Vector3f l2 = {v3->x - v2->x, v3->y - v2->y, v3->z - v2->z };

    // 외적을 통한 법선 벡터 계산.
    face.nx = l1.y * l2.z - l1.z * l2.y;
    face.ny = l1.z * l2.x - l1.x * l2.z;
    face.nz = l1.x * l2.y - l1.y * l2.x;

    // 정규화를 위한 법선 벡터의 크기 계산.
    float norm = sqrt(face.nx * face.nx + face.ny * face.ny + face.nz * face.nz);

    // 단위 법선 벡터 계산.
    face.nx /= norm;
    face.ny /= norm;
    face.nz /= norm;
}

__global__ void __calc_face_normals_cu(Face* faces, Vector3f* vertices, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    __calc_normal(faces[index], vertices);
}

void calc_face_normals(Object& obj) {
    for (Face& face : obj.faces) {
        __calc_normal(face, obj.vertices.data());
    }
}

void calc_face_normals_cu(Object& obj) {
    DeviceObject d_obj = {
        thrust::device_vector<Vector3f>(obj.vertices),
        thrust::device_vector<Face>(obj.faces)
    };

    size_t len_block = 1024;
    size_t len_grid = ceil((float)d_obj.faces.size() / len_block);

    __calc_face_normals_cu<<<len_grid, len_block>>>(thrust::raw_pointer_cast(d_obj.faces.data()),
                                                    thrust::raw_pointer_cast(d_obj.vertices.data()),
                                                    d_obj.faces.size());
    cudaDeviceSynchronize();

    thrust::copy(d_obj.vertices.begin(), d_obj.vertices.end(), obj.vertices.begin());
    thrust::copy(d_obj.faces.begin(), d_obj.faces.end(), obj.faces.begin());
    cudaDeviceSynchronize();
}
