## Include to trigger clang-format
if(BUILD_NO_CLANGFORMAT)
  return()
endif()

if(CLANGFORMAT_enabled)
  return()
endif()
set(CLANGFORMAT_enabled true)

set(CFMT_STYLE ${IPEX_PROJECT_TOP_DIR}/.clang-format)
if(NOT EXISTS ${CFMT_STYLE})
  message(WARNING "Cannot find style file ${CFMT_STYLE}!")
  return()
endif()

find_program(CLANG_FORMAT "clang-format")
if(NOT CLANG_FORMAT)
  message(WARNING "Please install clang-format before contributing to IPEX!")
endif()
