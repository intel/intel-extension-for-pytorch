cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-usm)

find_package(IPEX REQUIRED)

set(target example-usm)
add_executable(${target} example-usm.cpp)
target_link_libraries(${target} ${TORCH_IPEX_LIBRARIES})
list(APPEND CMAKE_CXX_FLAGS "-fsycl")

set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
