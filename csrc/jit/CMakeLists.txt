FILE(GLOB _JIT_SRCS *.cpp cpu/kernels/*.cpp passes/*.cpp codegen/*.cpp codegen/onednn/*.cpp cpu/tensorexpr/*.cpp cpu/tensorexpr/external_call/*.cpp)
LIST(APPEND IPEX_JIT_CPP_SRCS ${_JIT_SRCS})

# Pass to parent
set(IPEX_JIT_CPP_SRCS ${IPEX_JIT_CPP_SRCS} PARENT_SCOPE)
