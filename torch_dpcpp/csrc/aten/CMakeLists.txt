FILE(GLOB aten_functions_src "functions/*.cpp")
FILE(GLOB aten_core_src "core/*.cpp")

set(aten_src)
list(APPEND aten_src ${aten_functions_src})
list(APPEND aten_src ${aten_core_src})
set(DPCPP_ATEN_SRCS ${aten_src} PARENT_SCOPE)
