include(${IPEX_ROOT_DIR}/cmake/xpu/BuildFlags.cmake)

FILE(GLOB IPEX_PYTHON_CPP_XPU_SRCS *.cpp)
add_library(PY_XPU_OBJ OBJECT ${IPEX_PYTHON_CPP_XPU_SRCS})

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

target_include_directories(PY_XPU_OBJ PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(PY_XPU_OBJ PUBLIC ${IPEX_GPU_ROOT_DIR})
target_include_directories(PY_XPU_OBJ PUBLIC ${IPEX_GPU_ROOT_DIR}/aten)

target_include_directories(PY_XPU_OBJ PUBLIC ${PYTHON_INCLUDE_DIR})
set_target_properties(PY_XPU_OBJ PROPERTIES PYTHON_DIR ${PYTHON_INCLUDE_DIR})

target_include_directories(PY_XPU_OBJ PUBLIC ${pybind11_INCLUDE_DIRS})
set_target_properties(PY_XPU_OBJ PROPERTIES PYBIND11_DIR ${pybind11_INCLUDE_DIRS})

target_compile_definitions(PY_XPU_OBJ PUBLIC BUILD_WITH_XPU)

if(BUILD_SIMPLE_TRACE)
  target_compile_definitions(PY_XPU_OBJ PUBLIC BUILD_SIMPLE_TRACE)
endif()
