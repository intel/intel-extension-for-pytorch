cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(IPEX REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_IPEX_LIBRARIES}")

set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
