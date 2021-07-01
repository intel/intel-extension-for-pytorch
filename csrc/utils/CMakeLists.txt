FILE(GLOB utils_src "*.cpp")

set(IPEX_UTILS_SRCS ${utils_src} PARENT_SCOPE)

list(APPEND utils_headers
  ${IPEX_GPU_UTILS_SRC_DIR}/Env.h
  ${IPEX_GPU_UTILS_SRC_DIR}/Timer.h
  ${IPEX_GPU_UTILS_SRC_DIR}/Profiler.h
  ${IPEX_GPU_UTILS_SRC_DIR}/Settings.h
  ${IPEX_GPU_UTILS_SRC_DIR}/DPCPP.h
  ${IPEX_GPU_UTILS_SRC_DIR}/Macros.h)
set(IPEX_UTILS_HEADERS ${utils_headers} PARENT_SCOPE)
