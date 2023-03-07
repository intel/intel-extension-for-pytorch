cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(IPEX REQUIRED)

set(target example-app)
add_executable(${target} example-app.cpp)
target_link_libraries(${target} ${TORCH_IPEX_LIBRARIES})

set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
