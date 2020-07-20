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

cmake_minimum_required(VERSION 3.4.3)

#if(SYCL_cmake_included)
#    return()
#endif()
#set(SYCL_cmake_included true)
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

if(INTEL_SYCL_VERSION)
    get_filename_component(SYCL_INCLUDE_DIR
            "${INTEL_SYCL_VERSION}/../../.." ABSOLUTE)

    # Suppress the compiler warning about undefined CL_TARGET_OPENCL_VERSION
    add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

    find_library(SYCL_LIBRARY
        NAMES "sycl"
        HINTS
            ${sycl_root_hints}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
    if(NOT SYCL_LIBRARY)
        message(FATAL_ERROR "SYCL library not found")
    endif()

    # Find the OpenCL library from the SYCL distribution
    find_library(OpenCL_LIBRARY
        NAMES "OpenCL"
        HINTS
            ${sycl_root_hints}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH)
    if(NOT OpenCL_LIBRARY)
        message(FATAL_ERROR "OpenCL library not found")
    endif()
    set(OpenCL_INCLUDE_DIR ${SYCL_INCLUDE_DIR} CACHE STRING "")

    set(USE_DPCPP true)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_DPCPP" PARENT_SCOPE)
    add_definitions(-DUSE_DPCPP)
    if(USE_USM)
        add_definitions(-DUSE_USM)
    endif()

    message(STATUS "Intel SYCL include: ${SYCL_INCLUDE_DIR}")
    message(STATUS "Intel SYCL library: ${SYCL_LIBRARY}")
    message(STATUS "OpenCL include: ${OpenCL_INCLUDE_DIR}")
    message(STATUS "OpenCL library: ${OpenCL_LIBRARY}")

    if(NOT ${SYCL_INCLUDE_DIR} STREQUAL ${OpenCL_INCLUDE_DIR})
        include_directories(${OpenCL_INCLUDE_DIR})
    endif()

    include_directories(${SYCL_INCLUDE_DIR})

    list(APPEND EXTRA_SHARED_LIBS ${SYCL_LIBRARY})
    list(APPEND EXTRA_SHARED_LIBS ${OpenCL_LIBRARY})

    #add pstl lib
    # Try to find PSTL header from DPC++
    find_path(PSTL_INCLUDE_DIRS
            NAMES dpstd
            PATHS
            ${sycl_root_hints}
            PATH_SUFFIXES
            include
            NO_DEFAULT_PATH)

    find_package_handle_standard_args(PSTL
            FOUND_VAR PSTL_FOUND
            REQUIRED_VARS PSTL_INCLUDE_DIRS
            )

    #add tbb lib
    find_path(TBB_INCLUDE_DIRS
            NAMES tbb
            PATHS
                ${DPCPP_ROOT}/tbb/latest
                $ENV{DPCPP_ROOT}/tbb/latest
            PATH_SUFFIXES
                include
            NO_DEFAULT_PATH)

    find_package_handle_standard_args(TBB
            FOUND_VAR TBB_FOUND
            REQUIRED_VARS TBB_INCLUDE_DIRS
            )
    if(${TBB_FOUND})
        if(${PSTL_FOUND})
            add_definitions(-D_PSTL_BACKEND_SYCL)

            find_library(TBB_LIBRARY
                    NAMES tbb
                    HINTS
                        ${DPCPP_ROOT}/tbb/latest
                        $ENV{DPCPP_ROOT}/tbb/latest
                    PATH_SUFFIXES
                        lib/intel64/gcc4.8
                    NO_DEFAULT_PATH)
            if(NOT TBB_LIBRARY)
                message(FATAL_ERROR "TBB library not found")
            endif()

            list(APPEND EXTRA_SHARED_LIBS ${TBB_LIBRARY})

            if(NOT ${SYCL_INCLUDE_DIR} STREQUAL ${PSTL_INCLUDE_DIRS})
                include_directories(${PSTL_INCLUDE_DIRS})
            endif()

            if(NOT ${SYCL_INCLUDE_DIR} STREQUAL ${TBB_INCLUDE_DIRS})
                include_directories(${TBB_INCLUDE_DIRS})
            endif()

            MESSAGE(STATUS "TBB directors " ${TBB_INCLUDE_DIRS})
            MESSAGE(STATUS "PSTL directors " ${PSTL_INCLUDE_DIRS})
        else()
            MESSAGE(WARNING "PSTL not found. No PSTL")
        endif()
    else()
        MESSAGE(WARNING "TBB not found. No PSTL")
    endif()

else()
    # ComputeCpp-specific flags
    # 1. Ignore the warning about undefined symbols in SYCL kernels - comes from
    #    SYCL CPU thunks
    # 2. Fix remark [Computecpp:CC0027] about memcpy/memset intrinsics
    set(COMPUTECPP_USER_FLAGS
        -Wno-sycl-undef-func
        -no-serial-memop
        CACHE STRING "")
    set(ComputeCpp_DIR ${sycl_root_hint})
    include(cmake/Modules/FindComputeCpp.cmake)
    if(NOT ComputeCpp_FOUND)
        message(FATAL_ERROR "SYCL not found")
    endif()

    set(USE_COMPUTECPP true)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_COMPUTECPP" PARENT_SCOPE)
    add_definitions(-DUSE_COMPUTECPP)
    include_directories(SYSTEM ${ComputeCpp_INCLUDE_DIRS})
    list(APPEND EXTRA_SHARED_LIBS ${COMPUTECPP_RUNTIME_LIBRARY})

    include_directories(${OpenCL_INCLUDE_DIRS})
    list(APPEND EXTRA_SHARED_LIBS ${OpenCL_LIBRARIES})
endif()
