cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(torch_ipex VERSION "0.1.2")

set(IPEX_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${IPEX_ROOT_DIR}/cmake/Modules)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

set (CMAKE_GENERATOR "Unix Makefiles")

# set compilation env first
include(cmake/DPCPP.cmake)

# The options to build torch_ipex
include(CMakeDependentOption)
option(USE_ONEDPL "Use oneDPL" ON)
option(USE_ONEMKL "Use oneMKL BLAS" ON)
option(USE_LEVEL_ZERO_ONLY "Only enumerate devices with Level Zero" ON)
option(USE_PERSIST_STREAM "Use persistent oneDNN stream" ON)
option(USE_SCRATCHPAD_MODE "Use oneDNN scratchpad mode" ON)
option(USE_PRIMITIVE_CACHE "Cache oneDNN primitives by FRAMEWORK, instead of oneDNN itself" OFF)
option(USE_MULTI_CONTEXT "Create DPC++ runtime context per device" ON)
option(USE_ITT "Use Intel(R) VTune Profiler ITT functionality" ON)
option(BUILD_BY_PER_KERNEL "Build by DPC++ per_kernel option (exclusive with USE_AOT_DEVLIST)" OFF)
option(BUILD_NO_L0_ONEDNN "Build oneDNN without LevelZero support" OFF)
option(BUILD_STRIPPED_BIN "Strip all symbols after build" OFF)
option(BUILD_INTERNAL_DEBUG "Use internal debug code path" OFF)
cmake_dependent_option(BUILD_DOUBLE_KERNEL "Build double data type kernels" ON "BUILD_INTERNAL_DEBUG" OFF)
set(USE_AOT_DEVLIST "" CACHE STRING "Set device list for AOT build (for example, skl,ats,...)")

include(${PROJECT_SOURCE_DIR}/cmake/BuildOptions.cmake)

# Need the torch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(${IPEX_ROOT_DIR}/csrc)
add_subdirectory(${IPEX_ROOT_DIR}/torch_ipex/csrc)

function (print_configuration_summary)
  get_directory_property(CMAKE_COMPILE_DEFINITIONS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)

    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  CMake version         : ${CMAKE_VERSION}")
    message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Target name           : ${LIB_NAME}")
    message(STATUS "  Install path          : ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  Compile flags         : ${IPEX_COMPILE_FLAGS}")
    message(STATUS "  Compile definitions   : ${CMAKE_COMPILE_DEFINITIONS}")
    message(STATUS "  Linker options        : ${CMAKE_SHARED_LINKER_FLAGS}")

    message(STATUS "  Intel SYCL include    : ${SYCL_INCLUDE_DIR}")
    message(STATUS "  Intel SYCL library    : ${SYCL_LIBRARY}")
    message(STATUS "  OpenCL include        : ${OpenCL_INCLUDE_DIR}")
    message(STATUS "  OpenCL library        : ${OpenCL_LIBRARY}")
    message(STATUS "  LevelZero include     : ${LevelZero_INCLUDE_DIR}")
    message(STATUS "  LevelZero library     : ${LevelZero_LIBRARY}")
    message(STATUS "  USE_ONEDPL            : ${USE_ONEDPL}")

    message(STATUS "  USE_ONEMKL            : ${USE_ONEMKL}")
  if (USE_ONEMKL)
    message(STATUS "  ONEMKL include        : ${ONEMKL_INCLUDE_DIR}")
  endif(USE_ONEMKL)

    message(STATUS "  USE_LEVEL_ZERO_ONLY   : ${USE_LEVEL_ZERO_ONLY}")
    message(STATUS "  USE_PERSIST_STREAM    : ${USE_PERSIST_STREAM}")
    message(STATUS "  USE_PRIMITIVE_CACHE   : ${USE_PRIMITIVE_CACHE}")
    message(STATUS "  USE_SCRATCHPAD_MODE   : ${USE_SCRATCHPAD_MODE}")
    message(STATUS "  USE_MULTI_CONTEXT     : ${USE_MULTI_CONTEXT}")
    message(STATUS "  USE_ITT               : ${USE_ITT}")
  if (USE_ITT)
    message(STATUS "  ITT include           : ${ITT_INCLUDE_DIR}")
    message(STATUS "  ITT library           : ${ITT_LIBRARY}")
  endif(USE_ITT)
    message(STATUS "  BUILD_BY_PER_KERNEL   : ${BUILD_BY_PER_KERNEL}")
    message(STATUS "  BUILD_NO_L0_ONEDNN    : ${BUILD_NO_L0_ONEDNN}")
    message(STATUS "  BUILD_STRIPPED_BIN    : ${BUILD_STRIPPED_BIN}")

  if(NOT BUILD_BY_PER_KERNEL AND USE_AOT_DEVLIST)
    message(STATUS "  USE_AOT_DEVLIST       : ${USE_AOT_DEVLIST}")
  else()
    message(STATUS "  USE_AOT_DEVLIST       : OFF")
  endif()

    message(STATUS "  BUILD_INTERNAL_DEBUG  : ${BUILD_INTERNAL_DEBUG}")
  if (BUILD_INTERNAL_DEBUG)
    message(STATUS "  BUILD_DOUBLE_KERNEL   : ${BUILD_DOUBLE_KERNEL}")
  endif(BUILD_INTERNAL_DEBUG)

    message(STATUS "")
endfunction()

print_configuration_summary()
