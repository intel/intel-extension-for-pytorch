set(ISA_HELP_NAME "isa_help")

# ---[ Build flags
include(${IPEX_ROOT_DIR}/cmake/cpu/BuildFlags.cmake)

set(IPEX_CPU_CPP_ISA_SRCS)
add_subdirectory(${IPEX_CPU_ROOT_DIR}/isa ${ISA_HELP_NAME}_isa_code)

# Current Directory Sources
FILE(GLOB IPEX_ISA_HELP_SRCS *.cpp *.h)

set(ISA_HELP_SRCS ${IPEX_ISA_HELP_SRCS} ${IPEX_CPU_CPP_ISA_SRCS})

add_library(${ISA_HELP_NAME} ${ISA_HELP_SRCS})

target_include_directories(${ISA_HELP_NAME} PUBLIC ${IPEX_CPU_ROOT_DIR}/isa)

install(TARGETS ${ISA_HELP_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})