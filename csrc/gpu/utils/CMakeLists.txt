set(utils_src
    ${IPEX_GPU_ROOT_DIR}/utils/oneMKLUtils.cpp
    ${IPEX_GPU_ROOT_DIR}/utils/Settings.cpp
    ${IPEX_GPU_ROOT_DIR}/utils/SimpleTrace.cpp
    ${IPEX_GPU_ROOT_DIR}/utils/LogImpl.cpp)

if(BUILD_SYNGRAPH)
  list(APPEND utils_src ${IPEX_GPU_ROOT_DIR}/utils/logging_hl.cpp)
endif()

set(IPEX_UTILS_SRCS ${utils_src} PARENT_SCOPE)
