#include <stdio.h>

#include "dtypes.h"
#include "objloader.h"
#include "objutils.h"

int main() {
    std::string file_path = "";
    object_t obj = load_obj(file_path);

#if 1
    calc_face_normals(obj);
#else
    calc_face_normals_cu(obj);
#endif

#if 0
    printf("Vertices info:\n");
    for (vertex_t& vertex : obj.vertices) {
        printf("%f, %f, %f\n", vertex.x, vertex.y, vertex.z);
    }
    printf("\n");

    printf("Faces info:\n");
    for (face_t& face : obj.faces) {
        printf("%2zu, %2zu, %2zu  |  ", face.pi,  face.qi, face.ri );
        printf("%f, %f, %f\n", face.nx, face.ny, face.nz );
    }
#endif

    return 0;
}
