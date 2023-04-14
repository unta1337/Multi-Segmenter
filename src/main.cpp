#include "model.hpp"
#include "objutils.h"
#include "serialsegmenter.h"
#include <iostream>

int main() {
    Model model;
    model.read_obj("assets/Cube_noised.obj");

    std::cout << "Mesh 1 vertex size : " << model.meshes[0]->vertex.size()
              << std::endl;
    std::cout << "Mesh 1 triangle size : " << model.meshes[0]->index.size()
              << std::endl;

    SerialSegmenter serial_segmenter(model.meshes[0], 15.f);
    auto seg = serial_segmenter.do_segmentation();

    std::cout << "Segmentation result " << seg.size() << std::endl;
    for (auto& s : seg) {
        std::cout << s->name << std::endl;
        std::cout << s->vertex.size() << std::endl;
    }

    write_obj(seg, "", false);
    write_obj(seg, "Segmented_cube", true);

    return 0;
}
