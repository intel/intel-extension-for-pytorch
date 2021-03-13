# This file solely exists to let Caffe2 Android build get at the list
# of core files without having to trundle through all of ATen's CMakeLists.txt
# Add files needed from jit folders
LIST(APPEND DPCPP_COMMON_SRCS
    ${DPCPP_ROOT}/aten_ipex_bridge.cpp
    ${DPCPP_ROOT}/aten_ipex_type.cpp
    ${DPCPP_ROOT}/dpcpp_allocator.cpp
    ${DPCPP_ROOT}/init_python_bindings.cpp
    ${DPCPP_ROOT}/ipex_tensor_impl.cpp
    ${DPCPP_ROOT}/ipex_sparse_tensor_impl.cpp
    ${DPCPP_ROOT}/version.cpp
    ${DPCPP_ROOT}/utils.cpp
    ${DPCPP_ROOT}/distributed/xpu_ccl.cpp
    ${DPCPP_ROOT}/autocast_mode.cpp
)

# Pass to parent
set(DPCPP_COMMON_SRCS ${DPCPP_COMMON_SRCS} PARENT_SCOPE)
