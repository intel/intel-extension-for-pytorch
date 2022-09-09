cmake_minimum_required(VERSION 3.10)

# set the project name
project(cpu_feature)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Setup project top directory.
set(IPEX_PROJECT_TOP_DIR "${PROJECT_SOURCE_DIR}/../../../")
set(CPU_FEATURE_SRC "${IPEX_PROJECT_TOP_DIR}/csrc/cpu/isa/")

add_definitions (-DCPU_FEATURE_EXEC)
add_executable (cpu_features ${CPU_FEATURE_SRC}/cpu_feature.cpp ${CPU_FEATURE_SRC}/cpu_feature_main.cpp)
