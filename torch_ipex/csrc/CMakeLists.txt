include(${PROJECT_SOURCE_DIR}/cmake/BuildOptions.cmake)
#include(${PROJECT_SOURCE_DIR}/cmake/Codegen.cmake)

set(IPEX_C_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(IPEX_GPU_ROOT "${IPEX_C_SOURCE_DIR}/gpu")

set(TORCH_IPEX_SRCS)
set(IPEX_COMMON_SRCS)
set(IPEX_GPU_SRC)

FILE(GLOB IPEX_COMMON_SRCS "*.cpp")
list(REMOVE_ITEM IPEX_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/_C.cpp")

install(DIRECTORY "./" DESTINATION include FILES_MATCHING PATTERN "*.h")

add_subdirectory(${IPEX_GPU_ROOT})

list(APPEND TORCH_IPEX_SRCS ${IPEX_COMMON_SRCS})
add_library(torch_ipex SHARED ${TORCH_IPEX_SRCS})
set_target_properties(torch_ipex PROPERTIES PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")
set_target_properties(torch_ipex PROPERTIES OUTPUT_NAME ${LIB_NAME})

if (USE_ITT)
    find_package(ITT QUIET)
    if (ITT_FOUND)
        target_link_libraries(torch_ipex PUBLIC ${ITT_LIBRARY})
        target_include_directories(torch_ipex PUBLIC ${ITT_INCLUDE_DIR})
        target_compile_definitions(torch_ipex PUBLIC USE_ITT)

        set(DPCPP_ITT_SRCS)
        add_subdirectory(./itt)
        target_sources(torch_ipex PUBLIC ${DPCPP_ITT_SRCS})
    else()
        set(USE_ITT OFF)
        message(WARNING "Cannot find ITT.")
    endif()
endif()

set_target_properties(torch_ipex PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

target_link_libraries(torch_ipex PUBLIC torch_ipex_gpu)

install(TARGETS torch_ipex LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})