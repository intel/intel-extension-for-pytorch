cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(torch_ipex)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TORCH_IPEX_C_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/torch_ipex/csrc")

# The options to build torch_ipex
include(CMakeDependentOption)
option(DPCPP_ENABLE_PROFILING "Enable profiling for DPCPP" OFF)
cmake_dependent_option(DPCPP_KER_PRINT "Print kernel time" OFF "DPCPP_ENABLE_PROFILING" OFF)
option(USE_SYCL "Enable GPU sycl" ON)
cmake_dependent_option(USE_USM "Use USM" OFF "USE_SYCL" OFF)

include(cmake/GPU.cmake)

function (print_configuration_summary)
    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  CMake version         : ${CMAKE_VERSION}")
    message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  USE_SYCL              : ${USE_SYCL}")
    message(STATUS "  USE_USM               : ${USE_USM}")
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    get_directory_property( COMPILE_DEFINITIONS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
    message(STATUS "  Pre-definitions       : ${COMPILE_DEFINITIONS}")
endfunction()

print_configuration_summary()