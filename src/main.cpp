#include "model.hpp"
#include "obj_utils.h"
#include "serialsegmentor.h"
#include <iostream>

int main() {
    Model tmp;
    tmp.read_obj("assets/Cube_noised.obj");

    std::cout << "Mesh 1 vertex size : " << tmp.meshes[0]->vertex.size()
              << std::endl;
    std::cout << "Mesh 1 triangle size : " << tmp.meshes[0]->index.size()
              << std::endl;

    SerialSegmentor segs(tmp.meshes[0], 15.f);
    auto seg = segs.do_segmentation();

    std::cout << "Segmentation result " << seg.size() << std::endl;
    for (auto& s : seg) {
        std::cout << s->name << std::endl;
        std::cout << s->vertex.size() << std::endl;
    }

    write_obj(seg, "", false);
    write_obj(seg, "Segmented_cube", true);

    return 0;
}
