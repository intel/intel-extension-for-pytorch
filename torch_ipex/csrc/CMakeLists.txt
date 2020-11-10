add_subdirectory(./gpu)

FILE(GLOB TORCH_IPEX_MODULE_SRCS "py_init.cpp" "utils.cpp")

set(TORCH_IPEX_SRCS ${TORCH_IPEX_MODULE_SRCS} ${DPCPP_SRCS} PARENT_SCOPE)
