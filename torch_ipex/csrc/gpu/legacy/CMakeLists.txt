FILE(GLOB legacy_function_src "./*.cpp")

set(legacy_src)
list(APPEND legacy_src ${legacy_function_src})
set(DPCPP_LEGACY_SRCS ${legacy_src} PARENT_SCOPE)
