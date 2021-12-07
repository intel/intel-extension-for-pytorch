FILE(GLOB _CPU_SRCS *.cpp cpu/kernels/*.cpp cpu/passes/*.cpp codegen/*.cpp codegen/onednn/*.cpp)
LIST(APPEND DPCPP_JIT_SRCS ${_CPU_SRCS})

# Pass to parent
set(DPCPP_JIT_SRCS ${DPCPP_JIT_SRCS} PARENT_SCOPE)
