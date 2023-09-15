if(NOT BUILD_MODULE_TYPE STREQUAL "PYTHON")
  return()
endif()

include(${IPEX_ROOT_DIR}/cmake/BuildFlags.cmake)

set(CMAKE_CXX_STANDARD 17)

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

if (WINDOWS)
  # handling python linking
  # TODO: precise python finding for conda env
  find_package(Python COMPONENTS Interpreter Development)

  if(NOT DEFINED ${TORCH_PATH})
    # query torch path from python
    execute_process(COMMAND python -c "import sys; import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[0], file=sys.stderr)" ERROR_VARIABLE TORCH_FIRST_INCLUDE_DIR)
    get_filename_component(TORCH_ROOT ${TORCH_FIRST_INCLUDE_DIR}/../ ABSOLUTE)
    set(TORCH_PATH "${TORCH_ROOT}" CACHE FILEPATH "path to pytorch in the python installation")
    if(NOT (EXISTS ${TORCH_PATH}))
      message( FATAL_ERROR "Pytorch not found, is it not installed in the python distribution ${PYTHON_DIRECTORY}?")
    else()
      message(STATUS "Torch found at ${TORCH_PATH}")
    endif()
  endif(NOT DEFINED ${TORCH_PATH})
  message(STATUS "TORCH PATH: ${TORCH_PATH} ${TORCH_ROOT}")
  # add missing torch_python.lib
  set(TORCH_LIBRARIES ${TORCH_LIBRARIES} "${TORCH_ROOT}/lib/torch_python.lib")
endif(WINDOWS)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND11_CL_FLAGS}")

if(BUILD_WITH_CPU)
  add_subdirectory(cpu)
  list(APPEND PY_CPP_OBJS $<TARGET_OBJECTS:PY_CPU_OBJ>)
endif()

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
include_directories(${IPEX_CPU_CPP_THIRD_PARTY_ROOT}/libxsmm/include)

target_include_directories(intel-ext-pt-python PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(intel-ext-pt-python PUBLIC ${PYTHON_INCLUDE_DIR})
target_include_directories(intel-ext-pt-python PUBLIC ${pybind11_INCLUDE_DIRS})

target_link_directories(intel-ext-pt-python PRIVATE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

if(BUILD_STRIPPED_BIN)
  set_target_properties(intel-ext-pt-python PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

if(BUILD_WITH_CPU)
  target_compile_definitions(intel-ext-pt-python PUBLIC "BUILD_WITH_CPU")
  target_link_libraries(intel-ext-pt-python PUBLIC intel-ext-pt-cpu)
endif()

if(BUILD_WITH_XPU)
  target_compile_definitions(intel-ext-pt-python PUBLIC "BUILD_WITH_XPU")
  target_link_libraries(intel-ext-pt-python PUBLIC intel-ext-pt-gpu)
endif()

if(WINDOWS)
  target_link_libraries(intel-ext-pt-python PUBLIC ${Python_LIBRARIES})
  target_link_libraries(intel-ext-pt-python PUBLIC ${TORCH_LIBRARIES})
endif()

function (print_python_config_summary)
  get_target_property(PYTHON_LINK_LIBRARIES intel-ext-pt-python LINK_LIBRARIES)

  print_config_summary()
  message(STATUS "******** Summary on Python ********")
  message(STATUS "General:")

  message(STATUS "  C compiler            : ${CMAKE_C_COMPILER}")

  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler ID       : ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")

  message(STATUS "  CXX standard          : ${CMAKE_CXX_STANDARD}")
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  CXX Linker options    : ${CMAKE_SHARED_LINKER_FLAGS}")
  message(STATUS "  Link libraries        : ${PYTHON_LINK_LIBRARIES}")

  message(STATUS "  Torch version         : ${Torch_VERSION}")
  message(STATUS "  Torch include         : ${TORCH_INCLUDE_DIRS}")

  message(STATUS "  Python                : ${Python_FOUND}")
  message(STATUS "  Python version        : ${Python_VERSION}")
  message(STATUS "  Python development    : ${Python_Development_FOUND}")
  message(STATUS "  Python libraries      : ${Python_LIBRARIES}")  
  message(STATUS "  Python include        : ${PYTHON_INCLUDE_DIR}")
  message(STATUS "  pybind11 include      : ${pybind11_INCLUDE_DIRS}")

  message(STATUS "")
  message(STATUS "***********************************")
endfunction()

install(TARGETS intel-ext-pt-python
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

print_python_config_summary()
