FILE(GLOB _CPU_SRCS cpu/*.cpp cpu/optimizer/*.cpp cpu/utils/*.cpp)
FILE(GLOB_RECURSE _CPU_KERNELS_SRCS cpu/kernels/*.cpp)
LIST(APPEND DPCPP_ATEN_SRCS ${_CPU_SRCS} ${_CPU_KERNELS_SRCS})

# Pass to parent
set(DPCPP_ATEN_SRCS ${DPCPP_ATEN_SRCS} PARENT_SCOPE)
