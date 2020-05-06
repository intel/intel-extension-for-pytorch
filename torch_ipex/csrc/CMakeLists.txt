# This file solely exists to let Caffe2 Android build get at the list
# of core files without having to trundle through all of ATen's CMakeLists.txt
# Add files needed from jit folders
LIST(APPEND DPCPP_COMMON_SRCS
    ${TORCH_IPEX_C_SOURCE_DIR}/aten_ipex_bridge.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/aten_ipex_type_default.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/aten_ipex_sparse_type_default.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/aten_ipex_type_ext.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/aten_ipex_type.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/dpcpp_allocator.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/init_python_bindings.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/ipex_tensor_impl.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/ipex_sparse_tensor_impl.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/version.cpp
    ${TORCH_IPEX_C_SOURCE_DIR}/utils.cpp
)

# Pass to parent
set(DPCPP_COMMON_SRCS ${DPCPP_COMMON_SRCS} PARENT_SCOPE)
