if (SPDLOG_FOUND)
  return()
endif()

set(SPDLOG_FOUND OFF)
set(SPDLOG_INCLUDE_DIRS)

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
set(SPDLOG_ROOT "${THIRD_PARTY_DIR}/spdlog")

add_subdirectory(${SPDLOG_ROOT} build)

list(APPEND SPDLOG_INCLUDE_DIRS "${SPDLOG_ROOT}/include/")

set(SPDLOG_FOUND ON)
message(STATUS "Found spdlog: TRUE")