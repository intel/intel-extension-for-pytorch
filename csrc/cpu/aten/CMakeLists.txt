FILE(GLOB _CPU_SRCS *.cpp optimizer/*.cpp utils/*.cpp)
FILE(GLOB_RECURSE _CPU_KERNELS_SRCS kernels/*.cpp)
LIST(APPEND IPEX_CPU_CPP_ATEN_SRCS ${_CPU_SRCS})
# LIST(APPEND IPEX_CPU_CPP_ATEN_SRCS ${_CPU_KERNELS_SRCS})

# Pass to parent
set(IPEX_CPU_CPP_ATEN_SRCS ${IPEX_CPU_CPP_ATEN_SRCS} PARENT_SCOPE)
