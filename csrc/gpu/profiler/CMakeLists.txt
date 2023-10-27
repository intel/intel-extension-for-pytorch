set(profiler_src)
if(NOT USE_KINETO)
  list(APPEND profiler_src "${IPEX_GPU_PROFILER_SRC_DIR}/profiler_kineto.cpp")
  list(APPEND profiler_src "${IPEX_GPU_PROFILER_SRC_DIR}/kineto_shim.cpp")
else()
  FILE(GLOB profiler_src "*.cpp")
endif()

set(profiler_inc)
list(APPEND profiler_inc "${IPEX_GPU_PROFILER_SRC_DIR}/include")

set(IPEX_PROFILER_SRCS ${profiler_src} PARENT_SCOPE)
set(IPEX_PROFILER_INCLUDES ${profiler_inc} PARENT_SCOPE)
