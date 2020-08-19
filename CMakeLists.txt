cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(torch_ipex)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TORCH_IPEX_C_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_ipex/csrc")

include(cmake/DPCPP.cmake)

# The options to build torch_ipex
include(CMakeDependentOption)
cmake_dependent_option(USE_USM "Use USM" OFF "USE_DPCPP" OFF)

option(BUILD_INTERNAL_DEBUG "Use internal debug code path" ON)
cmake_dependent_option(
    BUILD_DOUBLE_KERNEL "Build double data type kernels" OFF "BUILD_INTERNAL_DEBUG" OFF)


include(cmake/GPU.cmake)

function (print_configuration_summary)
    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  CMake version         : ${CMAKE_VERSION}")
    message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Target name           : ${LIB_NAME}")
    message(STATUS "  Install path          : ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  USE_DPCPP             : ${USE_DPCPP}")
    message(STATUS "  USE_COMPUTECPP        : ${USE_COMPUTECPP}")
    message(STATUS "  USE_USM               : ${USE_USM}")
    message(STATUS "  BUILD_INTERNAL_DEBUG  : ${BUILD_INTERNAL_DEBUG}")
    message(STATUS "  BUILD_DOUBLE_KERNEL   : ${BUILD_DOUBLE_KERNEL}")
    message(STATUS "  ONEDNN_USE_SYCL       : ${ONEDNN_USE_SYCL}")
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    get_directory_property( COMPILE_DEFINITIONS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
    message(STATUS "  Pre-definitions       : ${COMPILE_DEFINITIONS}")
endfunction()

print_configuration_summary()
