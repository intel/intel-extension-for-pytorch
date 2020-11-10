add_subdirectory(./aten)
add_subdirectory(./jit)

FILE(GLOB aten_core_src "Module.cpp" "Stream.cpp")

set(DPCPP_SRCS ${aten_core_src} ${DPCPP_ATEN_SRCS} ${DPCPP_JIT_SRCS} PARENT_SCOPE)
