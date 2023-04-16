#include "consoleutils.h"
#include "model.h"
#include "objutils.h"
#include "originalsegmenter.h"
#include "serialsegmenter.h"
#include <iostream>

std::string file_path;
std::string folder_path;
std::string filename;

void init_file_path(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        file_path += std::string(argv[i]) + " ";
    }

    file_path.pop_back();
    std::replace(file_path.begin(), file_path.end(), '\\', '/');
    size_t separator_next_index = file_path.find_last_of('/') + 1;
    folder_path = file_path.substr(0, separator_next_index);
    filename = file_path.substr(separator_next_index, file_path.length());
    size_t extension_index = filename.find_last_of('.');
    if (extension_index != std::string::npos) {
        filename = filename.substr(0, extension_index);
    }
}

int main(int argc, char* argv[]) {
    INIT_CONSOLE();

    if (argc != 2) {
        std::cout << "Usage:\n";
        std::cout << "    " << argv[0] << " {obj_file_path}\n";

        return 1;
    }
    init_file_path(argc, argv);

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
    write_obj(seg, folder_path, false);

    // 한꺼번에 .obj 저장.
    write_obj(seg, folder_path + "Segmented_" + filename, true);

    return 0;
}
