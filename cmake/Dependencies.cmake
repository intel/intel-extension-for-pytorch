# DNNL
## find DPCPP env
include(cmake/DPCPP.cmake)
find_package(MKLDNN QUIET)
find_package(MKLDPCPP QUIET)
if(NOT MKLDNN_FOUND)
  message(FATAL_ERROR "Cannot find DNNL")
else()
  include_directories(BEFORE SYSTEM ${MKLDNN_INCLUDE_DIR})
endif()
