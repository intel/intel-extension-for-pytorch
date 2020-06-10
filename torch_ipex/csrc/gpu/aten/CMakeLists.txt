FILE(GLOB aten_functions_src "operators/*.cpp")
FILE(GLOB aten_core_src "py_init.cpp" "utils/*.cpp" "core/*.cpp" "core/detail/*.cpp")

set(PROJECT_ATEN_GEN "${PROJECT_SOURCE_DIR}/torch_ipex/csrc/gpu/aten/generated/ATen")
set(aten_generate_src
  # "${PROJECT_ATEN_GEN}/LegacyTHFunctionsSYCL.cpp"
  "${PROJECT_ATEN_GEN}/aten_ipex_type_default.cpp"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/core" "${CMAKE_CURRENT_SOURCE_DIR}/utils"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

set(aten_src)
list(APPEND aten_src ${aten_functions_src})
list(APPEND aten_src ${aten_core_src})
list(APPEND aten_src ${aten_generate_src})
set(DPCPP_ATEN_SRCS ${aten_src} PARENT_SCOPE)
