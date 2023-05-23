cmake_minimum_required(VERSION 3.9)

project(
        Multi-Segmenter-CUDA
        VERSION 0.1
        DESCRIPTION "3D Object Segmentation with CUDA Parallel Computing"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
set(ROOT_DIR ${PROJECT_SOURCE_DIR})
set(MAIN_FILE src/main.cpp)

file(GLOB PROJECT_SRC ${ROOT_DIR}/src/*)
file(GLOB PROJECT_HEADERS ${ROOT_DIR}/inc/*)
file(COPY ../assets DESTINATION ${CMAKE_BINARY_DIR})

list(FILTER PROJECT_SRC EXCLUDE REGEX .*${MAIN_FILE})
include_directories(${ROOT_DIR}/inc)

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-utf-8>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=\"-utf-8\">")

# MSVC에서 min/max를 매크로 상수로 정의하여 헤더 파일 포함 시 문제 발생.
# 참고: https://cimple.tistory.com/254
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-DNOMINMAX>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=\"-DNOMINMAX\">")

# 단계별 디버그 출력 활성화.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-DDO_STEP_LOG>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=\"-DDO_STEP_LOG\">")

# 타이머 결과 출력 활성화.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-DDO_TIME_LOG>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=\"-DDO_TIME_LOG\">")

find_package(OpenMP REQUIRED)

include(FetchContent)

FetchContent_Declare(
        tinyobjloader
        GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader
        GIT_TAG bca2719a11e688b85ce9af21dcb156f3d8b918bc
)
FetchContent_MakeAvailable(tinyobjloader)

add_executable(Multi-Segmenter-CUDA ${MAIN_FILE} ${PROJECT_SRC} ${PROJECT_HEADERS})
target_include_directories(Multi-Segmenter-CUDA PRIVATE ${glm_SOURCE_DIR})
target_link_libraries(Multi-Segmenter-CUDA PRIVATE OpenMP::OpenMP_CXX tinyobjloader)

set_target_properties(Multi-Segmenter-CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Multi-Segmenter-CUDA PROPERTY CUDA_ARCHITECTURES 86)

if(APPLE)
    set_property(TARGET Multi-Segmenter-CUDA PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
