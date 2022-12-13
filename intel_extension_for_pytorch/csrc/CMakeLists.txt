if(NOT BUILD_MODULE_TYPE STREQUAL "PYTHON")
  return()
endif()

include(${IPEX_ROOT_DIR}/cmake/BuildFlags.cmake)

find_package(pybind11)
if(NOT pybind11_FOUND)
    find_package(pybind11 CONFIG)
    if(NOT pybind11_FOUND)
        message(FATAL_ERROR "System pybind11 not found")
    else()
        message(STATUS "SYSTEM pybind11 found")
    endif()
else()
    message(STATUS "pybind11 found")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND11_CL_FLAGS}")

add_subdirectory(cpu)
list(APPEND PY_CPP_OBJS $<TARGET_OBJECTS:PY_CPU_OBJ>)

if(BUILD_WITH_XPU)
  add_subdirectory(xpu)
  list(APPEND PY_CPP_OBJS $<TARGET_OBJECTS:PY_XPU_OBJ>)
endif()

add_library(intel-ext-pt-python SHARED ${PY_CPP_OBJS} init_python_bindings.cpp)

include(${IPEX_ROOT_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_NATIVE_CSRC_FILES ${IPEX_FRONT_END_ROOT_DIR}/**.[ch] ${IPEX_FRONT_END_ROOT_DIR}/**.[ch]pp)
  add_custom_target(CL_FORMAT_FRONT_END_NATIVE_CSRC COMMAND ${CLANG_FORMAT_EXEC} -i -style=file ${ALL_NATIVE_CSRC_FILES})
  add_dependencies(intel-ext-pt-python CL_FORMAT_FRONT_END_NATIVE_CSRC)
endif()

target_include_directories(intel-ext-pt-python PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(intel-ext-pt-python PUBLIC ${PYTHON_INCLUDE_DIR})
target_include_directories(intel-ext-pt-python PUBLIC ${pybind11_INCLUDE_DIRS})

target_link_directories(intel-ext-pt-python PRIVATE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

if(BUILD_STRIPPED_BIN)
  set_target_properties(intel-ext-pt-python PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

target_link_libraries(intel-ext-pt-python PUBLIC intel-ext-pt-cpu)

if(BUILD_WITH_XPU)
  target_compile_definitions(intel-ext-pt-python PUBLIC "BUILD_WITH_XPU")
  target_link_libraries(intel-ext-pt-python PUBLIC intel-ext-pt-gpu)
endif()

function (print_python_config_summary)
  get_target_property(PYTHON_LINK_LIBRARIES intel-ext-pt-python LINK_LIBRARIES)

  print_config_summary()
  message(STATUS "******** Summary on Python ********")
  message(STATUS "General:")

  message(STATUS "  C compiler            : ${CMAKE_C_COMPILER}")

  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")

  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  CXX Linker options    : ${CMAKE_SHARED_LINKER_FLAGS}")
  message(STATUS "  Link libraries        : ${PYTHON_LINK_LIBRARIES}")

  message(STATUS "  Python include        : ${PYTHON_INCLUDE_DIR}")
  message(STATUS "  pybind11 include      : ${pybind11_INCLUDE_DIRS}")

  message(STATUS "")
  message(STATUS "***********************************")
endfunction()

install(TARGETS intel-ext-pt-python
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

print_python_config_summary()
