cmake_minimum_required(VERSION 3.15)

project(
        Multi-Segmenter
        VERSION 0.1
        DESCRIPTION "3D Object Segmentation with CUDA Parallel Computing"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
set(ROOT_DIR ${PROJECT_SOURCE_DIR})
set(MAIN_FILE src/main.cpp)

file(GLOB PROJECT_SRC ${ROOT_DIR}/src/*)
file(GLOB PROJECT_HEADERS ${ROOT_DIR}/inc/*)
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

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

# OpenMP 사용
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-openmp>)

find_package(OpenMP REQUIRED)
find_package(Thrust REQUIRED)
thrust_create_target(Thrust)

include(FetchContent)

FetchContent_Declare(
        tinyobjloader
        GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader
        GIT_TAG bca2719a11e688b85ce9af21dcb156f3d8b918bc
)
FetchContent_MakeAvailable(tinyobjloader)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
)

FetchContent_GetProperties(glm)
if (NOT glm_POPULATED)
    FetchContent_Populate(glm)
    set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
    add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
endif ()

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.0
)
include(GoogleTest)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(Multi-Segmenter ${MAIN_FILE} ${PROJECT_SRC} ${PROJECT_HEADERS})
target_include_directories(Multi-Segmenter PRIVATE ${glm_SOURCE_DIR})
target_link_libraries(Multi-Segmenter PRIVATE OpenMP::OpenMP_CXX Thrust tinyobjloader)
set_target_properties(Multi-Segmenter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Multi-Segmenter PROPERTY CUDA_ARCHITECTURES 86)

if (APPLE)
    set_property(TARGET Multi-Segmenter PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif ()

add_executable(Multi-Segmenter-Test ${PROJECT_TESTS} ${PROJECT_SRC} ${PROJECT_HEADERS})
target_include_directories(Multi-Segmenter-Test PRIVATE ${glm_SOURCE_DIR})
target_link_libraries(Multi-Segmenter-Test PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX Thrust tinyobjloader)
set_target_properties(Multi-Segmenter-Test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Multi-Segmenter-Test PROPERTY CUDA_ARCHITECTURES 86)