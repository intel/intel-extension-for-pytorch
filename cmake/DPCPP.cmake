#===============================================================================
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Manage SYCL-related compiler flags
#===============================================================================

if(DPCPP_cmake_included)
    return()
endif()
set(DPCPP_cmake_included true)

cmake_minimum_required(VERSION 3.4.3)

include(FindPackageHandleStandardArgs)

set(sycl_root_hint)
if(DEFINED DPCPP_ROOT)
    set(sycl_root_hint ${DPCPP_ROOT})
elseif(DEFINED ENV{DPCPP_ROOT})
    set(sycl_root_hint $ENV{DPCPP_ROOT})
endif()

set(sycl_root_hints)
if(sycl_root_hint)
    list(APPEND sycl_root_hints ${sycl_root_hint})
else()
    list(APPEND sycl_root_hints ${SYCL_BUNDLE_ROOT})
    list(APPEND sycl_root_hints $ENV{SYCL_BUNDLE_ROOT})
endif()

# Try to find Intel SYCL version.hpp header
find_file(INTEL_SYCL_VERSION
    NAMES version.hpp
    PATHS
        ${sycl_root_hints}
    PATH_SUFFIXES
        include/CL/sycl
        include/sycl/CL/sycl
        lib/clang/11.0.0/include/CL/sycl
        lib/clang/10.0.0/include/CL/sycl
        lib/clang/9.0.0/include/CL/sycl
        lib/clang/8.0.0/include/CL/sycl
    NO_DEFAULT_PATH)

if(NOT INTEL_SYCL_VERSION)
  message(FATAL_ERROR "Can NOT find SYCL file path!")
endif()

get_filename_component(SYCL_INCLUDE_DIR "${INTEL_SYCL_VERSION}/../../.." ABSOLUTE)

find_library(SYCL_LIBRARY
    NAMES "sycl"
    HINTS ${sycl_root_hints}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH)
if(NOT SYCL_LIBRARY)
    message(FATAL_ERROR "SYCL library not found")
endif()

# Find the OpenCL library from the SYCL distribution
find_library(OpenCL_LIBRARY
        NAMES "OpenCL"
        HINTS ${sycl_root_hints}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
set(OpenCL_INCLUDE_DIR ${SYCL_INCLUDE_DIR} CACHE STRING "")

# Find LevelZero
find_path(LevelZero_INCLUDE_DIR
        NAMES level_zero/ze_api.h
        PATH_SUFFIXES include)
find_library(LevelZero_LIBRARY
        NAMES ze_loader
        PATH_SUFFIXES x86_64_linux_gnu lib lib/x64 lib64)

set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl")
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -D__STRICT_ANSI__")
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-unnamed-lambda")
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fno-sycl-early-optimizations")
# Explicitly limit the index range (< Max int32) in kernel
# set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-id-queries-fit-in-int")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
if(BUILD_BY_PER_KERNEL)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_kernel")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl, -T ${PROJECT_SOURCE_DIR}/cmake/per_ker.ld")
elseif(USE_AOT_DEVLIST)
    set(SPIRV_OPT "spir64-unknown-unknown-sycldevice")
    set(AOT_ARCH_OPT "spir64_gen-unknown-unknown-sycldevice")
    set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-targets=${AOT_ARCH_OPT},${SPIRV_OPT}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_source")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-targets=${AOT_ARCH_OPT},${SPIRV_OPT}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Xsycl-target-backend=${AOT_ARCH_OPT}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} '-device ${USE_AOT_DEVLIST}'")
else()
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-device-code-split=per_source")
endif()

function(append_sycl_flag SRC_FILES SYCL_FLAGS)
    foreach(src_file IN LISTS ${SRC_FILES})
        get_source_file_property(CUR_FLAGS ${src_file} COMPILE_FLAGS)
        list(REMOVE_ITEM CUR_FLAGS "NOTFOUND")
        set(CUR_FLAGS "${CUR_FLAGS} ${SYCL_FLAGS}")
        set_source_files_properties(${src_file} COMPILE_FLAGS "${CUR_FLAGS}")
    endforeach()
endfunction()

message(STATUS "DPCPP found. Compiling with SYCL support")
