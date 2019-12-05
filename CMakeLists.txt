cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PLUGIN_NAME intel_pytorch_extension)

set(DPCPP_ROOT "${PROJECT_SOURCE_DIR}/torch_dpcpp/csrc")
set(DPCPP_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Common dependencies

add_subdirectory(${DPCPP_THIRD_PARTY_ROOT}/pybind11)
add_subdirectory(${DPCPP_THIRD_PARTY_ROOT}/mkl-dnn)

include(cmake/Dependencies.cmake)

IF(USE_SYCL)
  include(cmake/GPU.cmake)
ELSE()
  include(cmake/CPU.cmake)
ENDIF()
