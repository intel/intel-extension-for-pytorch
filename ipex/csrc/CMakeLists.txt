set(IPEX_GPU_SRCS)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
# FIXME: Do not set kernel flags, after remove oneDPL dependency in Storage.cpp
append_compile_flags(IPEX_GPU_SRCS ${IPEX_SYCL_KERNEL_FLAGS})

set(IPEX_COMMON_SRCS)
FILE(GLOB IPEX_COMMON_SRCS "*.cpp")
list(REMOVE_ITEM IPEX_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/_C.cpp")

set(TORCH_IPEX_SRCS)
list(APPEND TORCH_IPEX_SRCS ${IPEX_COMMON_SRCS} ${IPEX_GPU_SRCS})
# TODO: remove this flag after building host files with gcc/g++
append_compile_flags(TORCH_IPEX_SRCS "-fsized-deallocation")
add_library(ipex_python SHARED ${TORCH_IPEX_SRCS})

include(${PROJECT_SOURCE_DIR}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_IPEX_CSRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/**.[ch] ${CMAKE_CURRENT_SOURCE_DIR}/**.[ch]pp)
  add_custom_target(CL_FORMAT_IPEX_CSRC COMMAND clang-format -i -style=file ${ALL_IPEX_CSRC_FILES})
  add_dependencies(ipex_python CL_FORMAT_IPEX_CSRC)
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
target_include_directories(ipex_python PUBLIC ${pybind11_INCLUDE_DIRS})
target_link_directories(ipex_python PUBLIC ${SYCL_LIBRARY_DIR})

if (USE_ITT)
    target_compile_definitions(ipex_python PUBLIC USE_ITT)
    target_include_directories(ipex_python PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/itt)
    target_sources(ipex_python PUBLIC ${DPCPP_ITT_SRCS})
endif()

set_target_properties(ipex_python PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

if(BUILD_STRIPPED_BIN)
  set_target_properties(ipex_python PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

target_link_libraries(ipex_python PUBLIC ipex_gpu)

install(TARGETS ipex_python LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
