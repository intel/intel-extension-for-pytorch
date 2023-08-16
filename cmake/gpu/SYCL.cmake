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

if(IntelSYCL_cmake_included)
  return()
endif()
set(IntelSYCL_cmake_included true)

include(FindPackageHandleStandardArgs)

find_package(IntelSYCL REQUIRED)
if(NOT IntelSYCL_FOUND)
  message(FATAL_ERROR "Cannot find IntelSYCL compiler!")
endif()

# Try to find Intel SYCL version.hpp header
find_file(INTEL_SYCL_VERSION
    NAMES version.hpp
    PATHS
        ${SYCL_INCLUDE_DIR}
    PATH_SUFFIXES
        sycl
        sycl/CL
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

set(IGC_OCLOC_VERSION)
find_program(OCLOC_EXEC ocloc)
if(OCLOC_EXEC)
  set(drv_ver_file "${PROJECT_BINARY_DIR}/OCL_DRIVER_VERSION")
  file(REMOVE ${drv_ver_file})
  execute_process(COMMAND ${OCLOC_EXEC} query OCL_DRIVER_VERSION WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  if(EXISTS ${drv_ver_file})
    file(READ ${drv_ver_file} drv_ver_contents)
    string(STRIP ${drv_ver_contents} IGC_OCLOC_VERSION)
  endif()
endif()

# Find LevelZero
find_path(LevelZero_INCLUDE_DIR
        NAMES level_zero/ze_api.h
        PATH_SUFFIXES include)
find_library(LevelZero_LIBRARY
        NAMES ze_loader
        PATH_SUFFIXES x86_64_linux_gnu lib lib/x64 lib64)

set(LEVEL_ZERO_VERSION)
if(LevelZero_LIBRARY)
  get_filename_component(level_zero_lib ${LevelZero_LIBRARY} REALPATH)
  string(REGEX MATCHALL "so(\.[0-9]+)+$" lz_lib_str "${level_zero_lib}")
  string(REGEX MATCHALL "[0-9]+.*$" lz_lib_ver "${lz_lib_str}")
  string(STRIP ${lz_lib_ver} LEVEL_ZERO_VERSION)
endif()

# Find the OpenCL library from the SYCL distribution
# XXX: Fetch OpenCL for oneDNN only
find_library(OpenCL_LIBRARY
        NAMES "OpenCL"
        HINTS ${SYCL_LIBRARY_DIR}
        NO_DEFAULT_PATH)
set(OpenCL_INCLUDE_DIR ${SYCL_INCLUDE_DIR} CACHE STRING "")

# The fast-math will be enabled by default in ICPX.
# Refer to [https://clang.llvm.org/docs/UsersManual.html#cmdoption-fno-fast-math]
# 1. We enable below flags here to be warn about NaN and Infinity,
# which will be hidden by fast-math by default.
# 2. The associative-math in fast-math allows floating point
# operations to be reassociated, which will lead to non-deterministic results.
# 3. The approx-func allows certain math function calls (such as log, sqrt, pow, etc)
# to be replaced with an approximately equivalent set of instructions or
# alternative math function calls, which have great errors.
set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -fhonor-nans)
set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -fhonor-infinities)
set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -fno-associative-math)
set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -fno-approx-func)

# Explicitly limit the index range (< Max int32) in kernel
# set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -fsycl-id-queries-fit-in-int)

# Set compilation optimization level
if (BUILD_OPT_LEVEL)
  if("${BUILD_OPT_LEVEL}" STREQUAL "1" OR "${BUILD_OPT_LEVEL}" STREQUAL "0")
    set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -O${BUILD_OPT_LEVEL})
  else()
    message(WARNING "UNKNOWN BUILD_OPT_LEVEL ${BUILD_OPT_LEVEL}")
  endif()
endif()

# Fetch max processor count
include(ProcessorCount)
ProcessorCount(proc_cnt)
if ((DEFINED ENV{MAX_JOBS}) AND ("$ENV{MAX_JOBS}" LESS_EQUAL ${proc_cnt}))
  set(SYCL_MAX_PARALLEL_LINK_JOBS $ENV{MAX_JOBS})
else()
  set(SYCL_MAX_PARALLEL_LINK_JOBS ${proc_cnt})
endif()
set(IPEX_SYCL_LINK_FLAGS ${IPEX_SYCL_LINK_FLAGS} -fsycl-max-parallel-link-jobs=${SYCL_MAX_PARALLEL_LINK_JOBS})

if(BUILD_BY_PER_KERNEL)
  set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -fsycl-device-code-split=per_kernel)
endif()

# Set AOT targt list in link flags
if(USE_AOT_DEVLIST)
  set(IPEX_SYCL_LINK_FLAGS ${IPEX_SYCL_LINK_FLAGS} -fsycl-targets=spir64_gen,spir64)
endif()

# Make assert available in sycl kernel
if(USE_SYCL_ASSERT)
  set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -DSYCL_FALLBACK_ASSERT=1)
endif()

# Disable ITT annotation instrument in sycl kernel
if(NOT USE_ITT_ANNOTATION)
  set(IPEX_SYCL_KERNEL_FLAGS ${IPEX_SYCL_KERNEL_FLAGS} -fno-sycl-instrument-device-code)
endif()

# Handle huge binary issue for multi-target AOT build
if(NOT BUILD_SEPARATE_OPS)
  if(BUILD_BY_PER_KERNEL OR USE_AOT_DEVLIST)
    set(IPEX_SYCL_LINK_FLAGS ${IPEX_SYCL_LINK_FLAGS} -fsycl-link-huge-device-code)
  endif()
endif()

# WARNING: Append link flags in kernel flags before adding offline options
set(IPEX_SYCL_KERNEL_FLAGS ${SYCL_FLAGS} ${IPEX_SYCL_KERNEL_FLAGS} ${IPEX_SYCL_LINK_FLAGS})

# Use IGC auto large GRF option explicitly for current stage. The option is default in previous IGC.
# Before fully controlling large GRF setting (trade off concurrency and memory spill), we will keep it, let compiler to choose.
set(IPEX_OFFLINE_OPTIONS "${IPEX_OFFLINE_OPTIONS} -cl-intel-enable-auto-large-GRF-mode")

# If FP64 is unsupported on certain GPU arch, warning all kernels with double
# data type operations, and finish/return WITHOUT any computations.
set(IPEX_OFFLINE_OPTIONS "${IPEX_OFFLINE_OPTIONS} -cl-poison-unsupported-fp64-kernels")

set(IPEX_OFFLINE_OPTIONS "-options '${IPEX_OFFLINE_OPTIONS}'")

# Set AOT targt list in offline options
if(USE_AOT_DEVLIST)
  # WARNING: Do NOT change the order between AOT device list and options
  set(IPEX_OFFLINE_OPTIONS "-device ${USE_AOT_DEVLIST} ${IPEX_OFFLINE_OPTIONS}")
endif()

# WARNING: No more offline options after this line
if(NOT WINDOWS)
  set(IPEX_OFFLINE_OPTIONS -Xs ${IPEX_OFFLINE_OPTIONS})
else()
  set(IPEX_OFFLINE_OPTIONS /Xs ${IPEX_OFFLINE_OPTIONS})
endif()

# WARNING: Offline options must be appended at the end of link flags
set(IPEX_SYCL_LINK_FLAGS ${SYCL_FLAGS} ${IPEX_SYCL_LINK_FLAGS} ${IPEX_OFFLINE_OPTIONS})

message(STATUS "IntelSYCL found. Compiling with SYCL support")
