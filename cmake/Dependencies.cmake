# DNNL
set(USE_SYCL ON CACHE BOOL "enable SYCL")
find_package(MKLDNN QUIET)
if(NOT MKLDNN_FOUND)
  message(FATAL_ERROR "Cannot find DNNL")
else()
  include_directories(BEFORE SYSTEM ${MKLDNN_INCLUDE_DIR})
endif()

# PyTorch
# TODO: generate aten related files for now
# mute compilation. see CMakeLists
#
# set(USE_SYCL OFF CACHE BOOL "enable SYCL")
# set(USE_CUDA OFF CACHE BOOL "enable CUDA")
# set(USE_FBGEMM OFF CACHE BOOL "enable FBGEMM")
# set(USE_MKLDNN OFF CACHE BOOL "enable MKLDNN")
# set(USE_NNPACK OFF CACHE BOOL "enable NNPACK")
# set(USE_DISTRIBUTED OFF CACHE BOOL "enable DISTRIBUTED")
# set(BUILD_CAFFE2_OPS OFF CACHE BOOL "enable Caffe2 OPs")
# 
# set(DPCPP_DEP torch)
# add_subdirectory(third_party/pytorch)
