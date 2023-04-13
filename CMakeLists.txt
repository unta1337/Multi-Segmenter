cmake_minimum_required(VERSION 3.9)

project(
        Multi-Segmentor
        VERSION 0.1
        DESCRIPTION "3D Object Segmentation with Parallel Computing"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(ROOT_DIR ${PROJECT_SOURCE_DIR})
set(MAIN_FILE src/main.cpp)

file(GLOB PROJECT_SRC ${ROOT_DIR}/src/*)
file(GLOB PROJECT_HEADERS ${ROOT_DIR}/inc/*)
file(GLOB PROJECT_TESTS ${ROOT_DIR}/tests/*)

list(FILTER PROJECT_SRC EXCLUDE REGEX .*${MAIN_FILE})
include_directories(${ROOT_DIR}/inc)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

find_package(OpenMP REQUIRED)

include(FetchContent)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG		   0.9.9.2
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

add_executable(Multi-Segmentor ${MAIN_FILE} ${PROJECT_SRC} ${PROJECT_HEADERS})
target_include_directories(Multi-Segmentor PRIVATE ${glm_SOURCE_DIR})
target_link_libraries(Multi-Segmentor PRIVATE OpenMP::OpenMP_CXX)

add_executable(Multi-Segmentor-Test ${PROJECT_TESTS} ${PROJECT_SRC} ${PROJECT_HEADERS})
target_include_directories(Multi-Segmentor-Test PRIVATE ${glm_SOURCE_DIR})
target_link_libraries(Multi-Segmentor-Test PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
gtest_discover_tests(Multi-Segmentor-Test)
