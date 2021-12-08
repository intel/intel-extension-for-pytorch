FILE(GLOB _CPU_SRCS cpu/*.cpp cpu/optimizer/*.cpp cpu/utils/*.cpp)
LIST(APPEND DPCPP_ATEN_SRCS ${_CPU_SRCS})

# Pass to parent
set(DPCPP_ATEN_SRCS ${DPCPP_ATEN_SRCS} PARENT_SCOPE)
