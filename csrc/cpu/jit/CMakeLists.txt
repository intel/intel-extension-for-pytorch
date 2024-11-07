FILE(GLOB _JIT_SRCS *.cpp cpu/kernels/*.cpp passes/*.cpp codegen/*.cpp codegen/onednn/*.cpp)
LIST(APPEND IPEX_CPU_CPP_JIT_SRCS ${_JIT_SRCS})

# Pass to parent
set(IPEX_CPU_CPP_JIT_SRCS ${IPEX_CPU_CPP_JIT_SRCS} PARENT_SCOPE)
