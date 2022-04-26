set(IPEX_GPU_SRCS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)

set(IPEX_COMMON_SRCS)
FILE(GLOB IPEX_COMMON_SRCS "*.cpp")
list(REMOVE_ITEM IPEX_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/_C.cpp")

set(TORCH_IPEX_SRCS)
list(APPEND TORCH_IPEX_SRCS ${IPEX_COMMON_SRCS} ${IPEX_GPU_SRCS})
add_library(intel-ext-pt-python SHARED ${TORCH_IPEX_SRCS})

include(${PROJECT_SOURCE_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_IPEX_CSRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/**.[ch] ${CMAKE_CURRENT_SOURCE_DIR}/**.[ch]pp)
  add_custom_target(CL_FORMAT_IPEX_CSRC COMMAND clang-format -i -style=file ${ALL_IPEX_CSRC_FILES})
  add_dependencies(intel-ext-pt-python CL_FORMAT_IPEX_CSRC)
endif()

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

message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
target_include_directories(intel-ext-pt-python PUBLIC ${pybind11_INCLUDE_DIRS})

if(BUILD_SIMPLE_TRACE)
  target_compile_definitions(intel-ext-pt-python PUBLIC BUILD_SIMPLE_TRACE)
endif()

if (USE_ITT)
    target_compile_definitions(intel-ext-pt-python PUBLIC USE_ITT)
    target_include_directories(intel-ext-pt-python PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/itt)
    target_sources(intel-ext-pt-python PUBLIC ${DPCPP_ITT_SRCS})
endif()

set_target_properties(intel-ext-pt-python PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

if(BUILD_STRIPPED_BIN)
  set_target_properties(intel-ext-pt-python PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

target_link_libraries(intel-ext-pt-python PUBLIC intel-ext-pt-gpu)

install(TARGETS intel-ext-pt-python
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
