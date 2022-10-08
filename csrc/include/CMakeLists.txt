list(APPEND gpu_headers
  ${IPEX_GPU_INCLUDE_DIR}/Macros.h
  ${IPEX_GPU_INCLUDE_DIR}/Settings.h
  ${IPEX_GPU_INCLUDE_DIR}/Stream.h
  ${IPEX_GPU_INCLUDE_DIR}/Profiler.h)
set(IPEX_GPU_HEADERS ${gpu_headers} PARENT_SCOPE)
