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

find_package(IntelDPCPP REQUIRED)
if(NOT IntelDPCPP_FOUND)
  message(FATAL_ERROR "Cannot find IntelDPCPP compiler!")
endif()

# Try to find Intel SYCL version.hpp header
find_file(INTEL_SYCL_VERSION
    NAMES version.hpp
    PATHS
        ${SYCL_INCLUDE_DIR}
    PATH_SUFFIXES
        sycl
        sycl/CL/sycl
    NO_DEFAULT_PATH)

if(NOT INTEL_SYCL_VERSION)
  message(FATAL_ERROR "Can NOT find SYCL version file!")
endif()

set(SYCL_COMPILER_VERSION)
file(READ ${INTEL_SYCL_VERSION} version_contents)
string(REGEX MATCHALL "__SYCL_COMPILER_VERSION +[0-9]+" VERSION_LINE "${version_contents}")
list(LENGTH VERSION_LINE ver_line_num)
if (${ver_line_num} EQUAL 1)
  string(REGEX MATCHALL "[0-9]+" SYCL_COMPILER_VERSION "${VERSION_LINE}")
endif()

set(SYCL_DRIVER_VERSION)
find_program(OCLOC_EXEC ocloc)
if(OCLOC_EXEC)
  set(drv_ver_file "${PROJECT_BINARY_DIR}/OCL_DRIVER_VERSION")
  file(REMOVE ${drv_ver_file})
  execute_process(COMMAND ${OCLOC_EXEC} query OCL_DRIVER_VERSION WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  if(EXISTS ${drv_ver_file})
    file(READ ${drv_ver_file} drv_ver_contents)
    string(STRIP ${drv_ver_contents} SYCL_DRIVER_VERSION)
  endif()
endif()

# Find LevelZero
find_path(LevelZero_INCLUDE_DIR
        NAMES level_zero/ze_api.h
        PATH_SUFFIXES include)
find_library(LevelZero_LIBRARY
        NAMES ze_loader
        PATH_SUFFIXES x86_64_linux_gnu lib lib/x64 lib64)

set(SYCL_LEVEL_ZERO_VERSION)
if(LevelZero_LIBRARY)
  get_filename_component(level_zero_lib ${LevelZero_LIBRARY} REALPATH)
  string(REGEX MATCHALL "so(\.[0-9]+)+$" lz_lib_str "${level_zero_lib}")
  string(REGEX MATCHALL "[0-9]+.*$" lz_lib_ver "${lz_lib_str}")
  string(STRIP ${lz_lib_ver} SYCL_LEVEL_ZERO_VERSION)
endif()

# Find the OpenCL library from the SYCL distribution
# XXX: Fetch OpenCL for oneDNN only
find_library(OpenCL_LIBRARY
        NAMES "OpenCL"
        HINTS ${SYCL_LIBRARY_DIR}
        NO_DEFAULT_PATH)
set(OpenCL_INCLUDE_DIR ${SYCL_INCLUDE_DIR} CACHE STRING "")

set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} ${SYCL_FLAGS}")

# The fast-math will be enabled by default in ICX
# We enable below flags here to be warn about NaN and Infinity,
# which will be hidden by fast-math by default.
# The associative-math in fast-math allows floating point
# operations to be reassociated, which will lead to non-deterministic results.
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fhonor-nans")
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fhonor-infinities")
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fno-associative-math")

# Explicitly limit the index range (< Max int32) in kernel
# set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-id-queries-fit-in-int")

# Set compilation optimization level
if (BUILD_OPT_LEVEL)
  if("${BUILD_OPT_LEVEL}" STREQUAL "1" OR "${BUILD_OPT_LEVEL}" STREQUAL "0")
    set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -O${BUILD_OPT_LEVEL}")
  else()
    message(WARNING "UNKNOWN BUILD_OPT_LEVEL ${BUILD_OPT_LEVEL}")
  endif()
endif()

# Fetch max processor count
include(ProcessorCount)
ProcessorCount(proc_cnt)
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-max-parallel-link-jobs=${proc_cnt}")

# If FP64 is unsupported on certain GPU arch, warning all kernels with double
# data type operations, and finish/return WITHOUT any computations.
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -Xs '-options -cl-poison-unsupported-fp64-kernels'")

if(BUILD_BY_PER_KERNEL)
  set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-device-code-split=per_kernel")
endif()

# Set AOT targt list
if(USE_AOT_DEVLIST)
  set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-targets=spir64_gen,spir64")
  set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -Xs '-device ${USE_AOT_DEVLIST}'")
endif()

# Since 2016 Debian start using RUNPATH instead of normally RPATH, which gave the annoy effect that
# allow LD_LIBRARY_PATH to override dynamic linking path. Depends on intention of linking priority,
# change below for best outcome: disable, using RPATH, enable, using RUNPATH
set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -Wl,--disable-new-dtags")

if(BUILD_BY_PER_KERNEL)
  set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -Wl,-T ${PROJECT_SOURCE_DIR}/cmake/per_ker.ld")
elseif(USE_AOT_DEVLIST)
  if(NOT BUILD_SEPARATE_OPS)
    # Use customized link script to workaround huge binary issue for multi-target AOT build
    set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -Wl,-T ${PROJECT_SOURCE_DIR}/cmake/single_aot.ld")
  endif()
endif()

message(STATUS "DPCPP found. Compiling with SYCL support")
