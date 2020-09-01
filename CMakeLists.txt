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
option(USE_PERSIST_STREAM "Use persistent oneDNN stream" ON)
cmake_dependent_option(BUILD_BY_PER_KERNEL "Build by DPC++ per_kernel option" ON "USE_DPCPP" OFF)
option(BUILD_INTERNAL_DEBUG "Use internal debug code path" OFF)
cmake_dependent_option(BUILD_DOUBLE_KERNEL "Build double data type kernels" ON "BUILD_INTERNAL_DEBUG" OFF)

include(cmake/GPU.cmake)

function (print_configuration_summary)
  get_directory_property(CMAKE_COMPILE_DEFINITIONS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
  get_target_property(IPEX_COMPILE_DEFINITIONS torch_ipex COMPILE_DEFINITIONS)

    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  CMake version         : ${CMAKE_VERSION}")
    message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Target name           : ${LIB_NAME}")
    message(STATUS "  Install path          : ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  USE_COMPUTECPP        : ${USE_COMPUTECPP}")
    message(STATUS "  USE_DPCPP             : ${USE_DPCPP}")
  if (USE_DPCPP)
    message(STATUS "  USE_USM               : ${USE_USM}")
    message(STATUS "  USE_PERSIST_STREAM    : ${USE_PERSIST_STREAM}")
    message(STATUS "  BUILD_BY_PER_KERNEL   : ${BUILD_BY_PER_KERNEL}")
  endif()
    message(STATUS "  BUILD_INTERNAL_DEBUG  : ${BUILD_INTERNAL_DEBUG}")
  if (BUILD_INTERNAL_DEBUG)
    message(STATUS "  BUILD_DOUBLE_KERNEL   : ${BUILD_DOUBLE_KERNEL}")
  endif()
    message(STATUS "  ONEDNN_USE_SYCL       : ${ONEDNN_USE_SYCL}")
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  Compile flags         : ${IPEX_COMPILE_FLAGS}")
    message(STATUS "  Compile definitions   : ${CMAKE_COMPILE_DEFINITIONS} ${IPEX_COMPILE_DEFINITIONS}")
    message(STATUS "  Linker options        : ${CMAKE_SHARED_LINKER_FLAGS}")
endfunction()

print_configuration_summary()
