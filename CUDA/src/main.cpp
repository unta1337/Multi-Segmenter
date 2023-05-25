#include <stdio.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <omp.h>

#include "consoleutils.h"
#include "logutils.h"
#include "model.h"
#include "objutils.h"
#include "serialsegmenter.h"
#include "cudaheader.h"

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
#if 1
    printf("Hello, world!\n");

    #pragma omp parallel num_threads(4)
    {
        printf("Hello from OpenMP thread %d!\n", omp_get_thread_num());
    }

    int count = 10;
    float value = 3.14f;

    int* d_count;
    float* d_value;
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_value, sizeof(float));

    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice);

    void* args[] = { d_count, d_value };

    void** d_args;
    cudaMalloc(&d_args, 2 * sizeof(void*));

    cudaMemcpy(d_args, args, 2 * sizeof(void*), cudaMemcpyHostToDevice);

    kernel_call(foo_cuda, dim3(1, 1, 1), dim3(4, 1, 1), d_args);

    return 0;
#else
    INIT_CONSOLE();

    if (argc < 3) {
        std::cout << "Usage:\n";
        std::cout << "\t" << argv[0] << " "
                  << "[Mode (serial or cuda)] [Tolerance (Float, Optional)] [ObjFilePath]\n";

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
    } else if (mode == "cuda") {
        std::cout << "Cuda not implemented yet.\n";
        return 1;
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
#endif
}
