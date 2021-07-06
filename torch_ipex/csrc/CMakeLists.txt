set(IPEX_GPU_SRCS)
add_subdirectory(./gpu)
append_sycl_flag(IPEX_GPU_SRCS ${IPEX_SYCL_KERNEL_FLAGS})

set(IPEX_COMMON_SRCS)
FILE(GLOB IPEX_COMMON_SRCS "*.cpp")
list(REMOVE_ITEM IPEX_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/_C.cpp")

set(TORCH_IPEX_SRCS)
list(APPEND TORCH_IPEX_SRCS ${IPEX_COMMON_SRCS} ${IPEX_GPU_SRCS})
# TODO: remove this flag after building host files with gcc/g++
append_sycl_flag(TORCH_IPEX_SRCS "-fsized-deallocation")
add_library(ipex_python SHARED ${TORCH_IPEX_SRCS})

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

if (USE_ITT)
    find_package(ITT QUIET)
    if (ITT_FOUND)
        set(ITT_LIBRARY ${ITT_LIBRARY} PARENT_SCOPE)
        set(ITT_INCLUDE_DIR ${ITT_INCLUDE_DIR} PARENT_SCOPE)
        target_link_libraries(ipex_python PUBLIC ${ITT_LIBRARY})
        target_include_directories(ipex_python PUBLIC ${ITT_INCLUDE_DIR})
        target_compile_definitions(ipex_python PUBLIC USE_ITT)

        set(DPCPP_ITT_SRCS)
        add_subdirectory(./itt)
        target_sources(ipex_python PUBLIC ${DPCPP_ITT_SRCS})
    else()
        set(USE_ITT OFF PARENT_SCOPE)
        message(WARNING "Cannot find ITT! Continue to build without ITT!")
    endif()
endif()

set_target_properties(ipex_python PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

if(BUILD_STRIPPED_BIN)
  set_target_properties(ipex_python PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

target_link_libraries(ipex_python PUBLIC ipex_gpu)

install(TARGETS ipex_python LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
