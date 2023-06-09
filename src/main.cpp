#include "consoleutils.h"
#include "cudasegmenter.h"
#include "logutils.h"
#include "model.h"
#include "objutils.h"
#include "parallelsegmenter.h"
#include "serialsegmenter.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

std::string mode;
std::string file_path;
std::string folder_path;
std::string filename;
float tolerance = 15.f;
std::string tolerance_string;

void init_file_path(int argc, char* argv[]) {
    mode = argv[1];

    bool is_tolerance_exist = true;
    try {
        tolerance = std::stof(argv[2]);
    } catch (...) {
        is_tolerance_exist = false;
    }

    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << tolerance;
    tolerance_string = stream.str();

    for (int i = 2 + is_tolerance_exist; i < argc; ++i) {
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

    if (argc < 3) {
        std::cout << "Usage:\n";
        std::cout << "\t" << argv[0] << " "
                  << "[Mode (serial or parallel or cuda)] [Tolerance (Float, Optional)] [ObjFilePath]\n";

        return 1;
    }
    init_file_path(argc, argv);

    Model model;
    model.read_obj(file_path);

    // 오브젝트를 구성하는 정점 개수.
    std::cout << "Mesh 1 vertex size : " << model.meshes[0]->vertex.size() << std::endl;
    // 오브젝트를 구성하는 삼각형 개수.
    std::cout << "Mesh 1 triangle size : " << model.meshes[0]->index.size() << std::endl;

    std::unique_ptr<Segmenter> segmenter;

    if (mode == "serial") {
        segmenter = std::make_unique<SerialSegmenter>(model.meshes[0], tolerance);
    } else if (mode == "parallel") {
        segmenter = std::make_unique<ParallelSegmenter>(model.meshes[0], tolerance);
    } else if (mode == "cuda") {
        segmenter = std::make_unique<CUDASegmenter>(model.meshes[0], tolerance);
    }
    auto segments = segmenter->do_segmentation();

    // 구분된 부분별 출력.
    std::cout << "==================================================" << std::endl;
    std::cout << "Number of Segments: " << segments.size() << std::endl;
    for (auto& s : segments) {
        std::cout << "  - Segment: " << s->name << ", Size: " << s->vertex.size() << std::endl;
    }
    std::cout << "==================================================" << std::endl;

    TIME_LOG(segmenter->timer.printTimer());



    std::string log_path =
        folder_path + "Segmented_" + mode + "_" + tolerance_string + "_" + filename + ".txt";
    segmenter->timer.printToFile((char*)log_path.c_str());

    STEP_LOG(std::cout << "[Begin] Saving Result.\n");
    // 한꺼번에 .obj 저장.
    write_obj(segments, folder_path + "Segmented_" + mode + "_" + tolerance_string + "_" + filename, true);

    STEP_LOG(std::cout << "[End] Saving Resuilt.\n");

    return 0;
}
