cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(${IPEX_PROJ_NAME} VERSION ${CMAKE_PROJECT_VERSION})

set(IPEX_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(IPEX_CPU_ROOT_DIR ${IPEX_ROOT_DIR}/csrc/cpu)
set(IPEX_GPU_ROOT_DIR ${IPEX_ROOT_DIR}/csrc/gpu)
set(IPEX_FRONT_END_ROOT_DIR ${IPEX_ROOT_DIR}/intel_extension_for_pytorch/csrc)
list(APPEND CMAKE_MODULE_PATH ${IPEX_ROOT_DIR}/cmake/Modules)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# relocatable binary with automatically rpath of depend libraries
set(CMAKE_INSTALL_RPATH $ORIGIN)

include(CMakeDependentOption)
# General options:
option(BUILD_WITH_XPU "Build XPU backend implementation" ON)
option(BUILD_NO_CLANGFORMAT "Build without force clang-format" OFF)
option(BUILD_STATS "Count statistics for each component during build process" OFF)
option(BUILD_STRIPPED_BIN "Strip all symbols after build" OFF)

if (BUILD_STATS)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "time -v")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "time -v")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "time -v")
endif()

# Common options:
set(_CXX11_ABI_FLAG 0)
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(_CXX11_ABI_FLAG 1)
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${_CXX11_ABI_FLAG}")

if(DEFINED PYBIND11_CL_FLAGS)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND11_CL_FLAGS}")
else()
  message(FATAL_ERROR, "Cannot get pybind11 abi compiler flags")
endif()


# CPU options:
if(BUILD_MODULE_TYPE STREQUAL "CPU")
  include(${IPEX_ROOT_DIR}/cmake/cpu/BuildFlags.cmake)
endif()

# XPU options:
if(BUILD_MODULE_TYPE STREQUAL "GPU")
  include(${IPEX_ROOT_DIR}/cmake/xpu/Options.cmake)
endif()

# Need the torch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "")

message(STATUS "******** General Summary ********")
message(STATUS "General:")
message(STATUS "  CMake version         : ${CMAKE_VERSION}")
message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Platform              : ${PYTHON_PLATFORM_INFO}")
message(STATUS "  Target name           : ${IPEX_PROJ_NAME}")
message(STATUS "  Target version        : ${CMAKE_PROJECT_VERSION}")
message(STATUS "  Install path          : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
message(STATUS "Options:")
message(STATUS "  BUILD_WITH_XPU        : ${BUILD_WITH_XPU}")
message(STATUS "  BUILD_NO_CLANGFORMAT  : ${BUILD_NO_CLANGFORMAT}")
message(STATUS "  BUILD_STATS           : ${BUILD_STATS}")
message(STATUS "  BUILD_STRIPPED_BIN    : ${BUILD_STRIPPED_BIN}")

message(STATUS "")

# Dispatch for build.
if(BUILD_MODULE_TYPE STREQUAL "CPU" OR BUILD_MODULE_TYPE STREQUAL "GPU")
  add_subdirectory(${IPEX_ROOT_DIR}/csrc)
endif()

if(BUILD_MODULE_TYPE STREQUAL "PYTHON")
 add_subdirectory(${IPEX_ROOT_DIR}/intel_extension_for_pytorch/csrc)
endif()
