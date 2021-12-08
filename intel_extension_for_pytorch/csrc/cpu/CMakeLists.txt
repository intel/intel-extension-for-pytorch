FILE(GLOB _CPU_SRCS *.cpp vec512/*.cpp ideep/*.cpp runtime/*.cpp isa/*.cpp)
LIST(APPEND DPCPP_CPU_SRCS ${_CPU_SRCS})

# Pass to parent
set(DPCPP_CPU_SRCS ${DPCPP_CPU_SRCS} PARENT_SCOPE)
