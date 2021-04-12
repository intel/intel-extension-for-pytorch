FILE(GLOB _CPU_SRCS *.cpp dbl/*.cpp int8/*.cpp bf16/*.cpp aten/operators/*.cpp toolkit/*.cpp)
LIST(APPEND DPCPP_CPU_SRCS ${_CPU_SRCS})

# Pass to parent
set(DPCPP_CPU_SRCS ${DPCPP_CPU_SRCS} PARENT_SCOPE)
