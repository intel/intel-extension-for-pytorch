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
        include
        include/CL/sycl
        include/sycl/CL/sycl
    NO_DEFAULT_PATH)

if(NOT INTEL_SYCL_VERSION)
  message(FATAL_ERROR "Can NOT find SYCL file path!")
endif()

set(SYCL_COMPILER_VERSION)
file(READ ${INTEL_SYCL_VERSION} version_contents)
string(REGEX MATCHALL "__SYCL_COMPILER_VERSION +[0-9]+" VERSION_LINE "${version_contents}")
list(LENGTH VERSION_LINE ver_line_num)
if (${ver_line_num} EQUAL 1)
  string(REGEX MATCHALL "[0-9]+" SYCL_COMPILER_VERSION "${VERSION_LINE}")
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

set(SYCL_DRIVER_VERSION)
find_program(OCLOC_EXEC ocloc)
if(OCLOC_EXEC)
  set(drv_ver_path "${IPEX_ROOT_DIR}/csrc/aten/generated")
  set(drv_ver_file "${drv_ver_path}/OCL_DRIVER_VERSION")
  file(REMOVE ${drv_ver_file})
  execute_process(COMMAND ${OCLOC_EXEC} query OCL_DRIVER_VERSION WORKING_DIRECTORY ${drv_ver_path})
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

set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl")
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -D__STRICT_ANSI__")
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-unnamed-lambda")
set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-early-optimizations")
# Explicitly limit the index range (< Max int32) in kernel
# set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-id-queries-fit-in-int")

set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -fsycl")
if(BUILD_BY_PER_KERNEL)
    set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -fsycl-device-code-split=per_kernel")
    set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -Wl, -T ${PROJECT_SOURCE_DIR}/cmake/per_ker.ld")
elseif(USE_AOT_DEVLIST)
    set(SPIRV_OPT "spir64-unknown-unknown-sycldevice")
    set(AOT_ARCH_OPT "spir64_gen-unknown-unknown-sycldevice")
    set(IPEX_SYCL_KERNEL_FLAGS "${IPEX_SYCL_KERNEL_FLAGS} -fsycl-targets=${AOT_ARCH_OPT},${SPIRV_OPT}")
    set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -fsycl-device-code-split=per_source")
    set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -fsycl-targets=${AOT_ARCH_OPT},${SPIRV_OPT}")
    set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -Xsycl-target-backend=${AOT_ARCH_OPT}")
    # FIXME: Provide revision ID to IGC for PVC platform to avoid AOT bug, only one device is passed if pvc in USE_AOT_DEVLIST
    string(REGEX MATCHALL "[a-zA-Z0-9]+" DEV_LIST "${USE_AOT_DEVLIST}")
    if("pvc" IN_LIST DEV_LIST OR "0xbd5" IN_LIST DEV_LIST)
        list(LENGTH DEV_LIST length)
        if (NOT ${length} EQUAL 1)
          message(FATAL_ERROR "Cannot enable AOT for multiple devices if PVC is required in the device list!")
        endif()
        set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} '-device ${USE_AOT_DEVLIST} -revision_id 3'")
    else()
        set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} '-device ${USE_AOT_DEVLIST}'")
    endif()
else()
    # Use auto mode of device code split
    set(IPEX_SYCL_LINKER_FLAGS "${IPEX_SYCL_LINKER_FLAGS} -fsycl-device-code-split")
endif()

message(STATUS "DPCPP found. Compiling with SYCL support")
