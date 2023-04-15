#include "consoleutils.h"
#include "model.h"
#include "objutils.h"
#include "originalsegmenter.h"
#include "serialsegmenter.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage:\n";
        std::cout << "    " << argv[0] << " {obj_file_path}\n";

        return 1;
    }

    std::string file_path = argv[1];

    INIT_CONSOLE();

    Model model;
    model.read_obj(file_path);

    // 오브젝트를 구성하는 정점 개수.
    std::cout << "Mesh 1 vertex size : " << model.meshes[0]->vertex.size() << std::endl;
    // 오브젝트를 구성하는 삼각형 개수.
    std::cout << "Mesh 1 triangle size : " << model.meshes[0]->index.size() << std::endl;

    SerialSegmenter serial_segmenter(model.meshes[0], 15.f);
    auto seg = serial_segmenter.do_segmentation();

    // 구분된 부분별 출력.
    std::cout << "Segmentation result " << seg.size() << std::endl;
    for (auto& s : seg) {
        std::cout << s->name << std::endl;
        std::cout << s->vertex.size() << std::endl;
    }

    // 구분된 부분별 .obj 저장. 각 부분별 명칭으로 저장됨. (i.e., cube_seg0.obj)
    write_obj(seg, "", false);

    // 한꺼번에 .obj 저장.
    write_obj(seg, "Segmented_cube", true);

    return 0;
}
