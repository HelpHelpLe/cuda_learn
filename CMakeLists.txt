cmake_minimum_required(VERSION 3.15)

project(cuda_learn)
enable_language(CUDA)

add_compile_options(-g)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(CUDA_SRC_DIR ${SRC_DIR}/cuda)

file(GLOB_RECURSE CC_FILES "${SRC_DIR}/*.cc")

file(GLOB_RECURSE CU_FILES "${CUDA_SRC_DIR}/*.cu")

add_executable(test ${CC_FILES} ${CU_FILES})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/)
