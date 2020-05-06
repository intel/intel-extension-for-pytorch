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

set(mkl_dpcpp_root_hint)
if(DEFINED MKL_DPCPP_ROOT)
    set(mkl_dpcpp_root_hint ${MKL_DPCPP_ROOT})
elseif(DEFINED ENV{MKL_DPCPP_ROOT})
    set(mkl_dpcpp_root_hint $ENV{MKL_DPCPP_ROOT})
endif()

# Try to find Intel MKL DPCPP header
find_file(MKL_DPCPP_HEADER
    NAMES mkl_sycl.hpp
    PATHS
        ${mkl_dpcpp_root_hint}
    PATH_SUFFIXES
        include
    NO_DEFAULT_PATH)

if(MKL_DPCPP_HEADER)
    get_filename_component(MKL_DPCPP_INCLUDE_DIR
            "${MKL_DPCPP_HEADER}/.." ABSOLUTE)

    find_library(MKL_DPCPP_LIBRARY
        NAMES "mkl_sycl"
        HINTS
            ${mkl_dpcpp_root_hint}
        PATH_SUFFIXES lib
                      lib/intel64
        NO_DEFAULT_PATH)
    if(NOT SYCL_LIBRARY)
        message(FATAL_ERROR "DPCPP library mkl_sycl not found")
    endif()

    get_filename_component(MKL_DPCPP_LIBRARY_DIR
              "${MKL_DPCPP_LIBRARY}/.." ABSOLUTE)

    message(STATUS "Intel MKL DPCPP include dir: ${MKL_DPCPP_INCLUDE_DIR}")
    message(STATUS "Intel MKL DPCPP library: ${MKL_DPCPP_LIBRARY}")

    add_definitions(-DUSE_MKL_SYCL)

    include_directories(${MKL_DPCPP_INCLUDE_DIR})

    list(APPEND EXTRA_SHARED_LIBS ${MKL_DPCPP_LIBRARY})
    list(APPEND EXTRA_SHARED_LIBS ${MKL_DPCPP_LIBRARY_DIR}/libmkl_intel_ilp64.so)
    list(APPEND EXTRA_SHARED_LIBS ${MKL_DPCPP_LIBRARY_DIR}/libmkl_core.so)
    list(APPEND EXTRA_SHARED_LIBS ${MKL_DPCPP_LIBRARY_DIR}/libmkl_sequential.so)
else()
    message(FATAL_ERROR "Intel MKL DPCPP Header not found")
endif()
