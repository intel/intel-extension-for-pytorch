# This file solely exists to let Caffe2 Android build get at the list
# of core files without having to trundle through all of ATen's CMakeLists.txt
# Add files needed from jit folders
LIST(APPEND DPCPP_COMMON_SRCS
    ${DPCPP_ROOT}/aten_ipex_bridge.cpp
    ${DPCPP_ROOT}/aten_ipex_type_default.cpp
    ${DPCPP_ROOT}/aten_ipex_type.cpp
    ${DPCPP_ROOT}/init_python_bindings.cpp
    ${DPCPP_ROOT}/tensor_impl.cpp
    ${DPCPP_ROOT}/version.cpp
)

# Pass to parent
set(DPCPP_COMMON_SRCS ${DPCPP_COMMON_SRCS} PARENT_SCOPE)
