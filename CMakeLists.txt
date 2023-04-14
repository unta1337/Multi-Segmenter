cmake_minimum_required(VERSION 3.9)

project(
        Multi-Segmenter
        VERSION 0.1
        DESCRIPTION "3D Object Segmentation with Parallel Computing"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(ROOT_DIR ${PROJECT_SOURCE_DIR})
set(MAIN_FILE src/main.cpp)

file(GLOB PROJECT_SRC ${ROOT_DIR}/src/*)
file(GLOB PROJECT_HEADERS ${ROOT_DIR}/inc/*)
file(GLOB PROJECT_TESTS ${ROOT_DIR}/tests/*)
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

list(FILTER PROJECT_SRC EXCLUDE REGEX .*${MAIN_FILE})
include_directories(${ROOT_DIR}/inc)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

find_package(OpenMP REQUIRED)

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
        GIT_TAG		   0.9.9.8
)

FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
    FetchContent_Populate(glm)
    set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
    add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
endif()

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
target_link_libraries(Multi-Segmenter PRIVATE OpenMP::OpenMP_CXX tinyobjloader)

add_executable(Multi-Segmenter-Test ${PROJECT_TESTS} ${PROJECT_SRC} ${PROJECT_HEADERS})
target_include_directories(Multi-Segmenter-Test PRIVATE ${glm_SOURCE_DIR})
target_link_libraries(Multi-Segmenter-Test PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX tinyobjloader)
gtest_discover_tests(Multi-Segmenter-Test)
