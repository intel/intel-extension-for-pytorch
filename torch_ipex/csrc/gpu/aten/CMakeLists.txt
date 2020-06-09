FILE(GLOB aten_functions_src "operators/*.cpp")
FILE(GLOB aten_core_src "py_init.cpp" "utils/*.cpp" "core/*.cpp" "core/detail/*.cpp")

set(PROJECT_ATEN_GEN "${PROJECT_SOURCE_DIR}/torch_ipex/csrc/gpu/aten/generated/ATen")
set(aten_generate_src
  # "${PROJECT_ATEN_GEN}/LegacyTHFunctionsSYCL.cpp"
  "${PROJECT_ATEN_GEN}/aten_ipex_type_default.cpp"
)

FILE(GLOB GPU_ATEN_CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h")
set(TORCH_IPEX_PUBLIC_HEADER ${GPU_ATEN_CORE_HEADER} ${TORCH_IPEX_PUBLIC_HEADER} PARENT_SCOPE)

set(aten_src)
list(APPEND aten_src ${aten_functions_src})
list(APPEND aten_src ${aten_core_src})
list(APPEND aten_src ${aten_generate_src})
set(DPCPP_ATEN_SRCS ${aten_src} PARENT_SCOPE)
