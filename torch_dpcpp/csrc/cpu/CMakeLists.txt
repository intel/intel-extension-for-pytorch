# This file solely exists to let Caffe2 Android build get at the list
# of core files without having to trundle through all of ATen's CMakeLists.txt
# Add files needed from jit folders
LIST(APPEND DPCPP_CPU_SRCS
    ${DPCPP_ROOT}/cpu/DPCPPAllocator.cpp
)

# Pass to parent
set(DPCPP_CPU_SRCS ${DPCPP_CPU_SRCS} PARENT_SCOPE)
