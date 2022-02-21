cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(IPEX CXX C)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PLUGIN_NAME intel-ext-pt-cpu)

set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}/lib/:${RPATH_VALUE}/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IPEX_INSTALL_LIBDIR})
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
SET(AVX_VERSION ${IPEX_AVX_VERSION})

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++14. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(_CXX11_ABI_FLAG 0)
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(_CXX11_ABI_FLAG 1)
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${_CXX11_ABI_FLAG}")

set(DPCPP_ROOT "${PROJECT_SOURCE_DIR}/intel_extension_for_pytorch/csrc")
set(DPCPP_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Common dependencies
include(cmake/CpuDynDisp.cmake)