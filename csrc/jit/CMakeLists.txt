FILE(GLOB _JIT_SRCS *.cpp)
LIST(APPEND IPEX_JIT_COMMON_CPP_SRCS ${_JIT_SRCS})

# Pass to parent
set(IPEX_JIT_COMMON_CPP_SRCS ${IPEX_JIT_COMMON_CPP_SRCS} PARENT_SCOPE)
