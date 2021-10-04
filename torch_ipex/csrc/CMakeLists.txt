# This file solely exists to let Caffe2 Android build get at the list
# of core files without having to trundle through all of ATen's CMakeLists.txt
# Add files needed from jit folders
LIST(APPEND DPCPP_COMMON_SRCS
    ${DPCPP_ROOT}/version.cpp
    ${DPCPP_ROOT}/utils.cpp
    ${DPCPP_ROOT}/verbose.cpp
    ${DPCPP_ROOT}/autocast_mode.cpp
    ${DPCPP_ROOT}/LlgaTensorImpl.cpp
    ${DPCPP_ROOT}/autocast_kernel.cpp
    ${DPCPP_ROOT}/autocast_verbose.cpp
)

# Pass to parent
set(DPCPP_COMMON_SRCS ${DPCPP_COMMON_SRCS} PARENT_SCOPE)
