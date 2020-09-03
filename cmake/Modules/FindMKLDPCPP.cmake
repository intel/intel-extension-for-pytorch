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

# The following are set after configuration is done:
#  ONEMKL_FOUND            : set to true if oneMKL is found.
#  ONEMKL_INCLUDE_DIR      : path to oneMKL include dir.
#  ONEMKL_SHARED_LIBS      : list of libraries for oneMKL
#===============================================================================

cmake_minimum_required(VERSION 3.4.3)

if (NOT MKLDPCPP_FOUND)
set(MKLDPCPP_FOUND OFF)

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
    get_filename_component(ONEMKL_INCLUDE_DIR
            "${MKL_DPCPP_HEADER}/.." ABSOLUTE)

    find_library(ONEMKL_SYCL_LIBRARY
        NAMES "mkl_sycl"
        HINTS
            ${mkl_dpcpp_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_SYCL_LIBRARY)
        message(FATAL_ERROR "DPCPP library mkl_sycl not found")
    endif()

    find_library(ONEMKL_ILP64_LIBRARY
        NAMES "mkl_intel_ilp64"
        HINTS
            ${mkl_dpcpp_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_ILP64_LIBRARY)
        message(FATAL_ERROR "DPCPP library mkl_intel_ilp64 not found")
    endif()

    find_library(ONEMKL_CORE_LIBRARY
        NAMES "mkl_core"
        HINTS
            ${mkl_dpcpp_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_CORE_LIBRARY)
        message(FATAL_ERROR "DPCPP library mkl_core not found")
    endif()

    find_library(ONEMKL_THREADS_LIBRARY
        NAMES "mkl_sequential"
        HINTS
            ${mkl_dpcpp_root_hint}
        PATH_SUFFIXES
            lib
            lib/intel64
        NO_DEFAULT_PATH)
    if(NOT ONEMKL_THREADS_LIBRARY)
        message(FATAL_ERROR "DPCPP library mkl_sequential not found")
    endif()

    message(STATUS "Intel MKL DPCPP include dir: ${ONEMKL_INCLUDE_DIR}")
    message(STATUS "Intel MKL DPCPP library: ${ONEMKL_SYCL_LIBRARY}")
    message(STATUS "Intel MKL DPCPP library: ${ONEMKL_ILP64_LIBRARY}")
    message(STATUS "Intel MKL DPCPP library: ${ONEMKL_CORE_LIBRARY}")
    message(STATUS "Intel MKL DPCPP library: ${ONEMKL_THREADS_LIBRARY}")

    list(APPEND ONEMKL_SHARED_LIBS ${ONEMKL_SYCL_LIBRARY} ${ONEMKL_ILP64_LIBRARY} ${ONEMKL_CORE_LIBRARY} ${ONEMKL_THREADS_LIBRARY})
    set(MKLDPCPP_FOUND ON)
    message(STATUS "Intel oneMKL found.")
else()
  message(WARNING "Intel oneMKL not found. No DPCPP MKL support")
endif()


endif()
