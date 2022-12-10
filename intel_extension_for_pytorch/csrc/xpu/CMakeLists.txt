include(${IPEX_ROOT_DIR}/cmake/gpu/BuildFlags.cmake)

FILE(GLOB IPEX_PYTHON_CPP_XPU_SRCS *.cpp)
add_library(PY_XPU_OBJ OBJECT ${IPEX_PYTHON_CPP_XPU_SRCS})

target_include_directories(PY_XPU_OBJ PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(PY_XPU_OBJ PUBLIC ${IPEX_GPU_ROOT_DIR})
target_include_directories(PY_XPU_OBJ PUBLIC ${IPEX_GPU_ROOT_DIR}/aten)

target_include_directories(PY_XPU_OBJ PUBLIC ${PYTHON_INCLUDE_DIR})
target_include_directories(PY_XPU_OBJ PUBLIC ${pybind11_INCLUDE_DIRS})

target_compile_definitions(PY_XPU_OBJ PUBLIC BUILD_WITH_XPU)

if(BUILD_SIMPLE_TRACE)
  target_compile_definitions(PY_XPU_OBJ PUBLIC BUILD_SIMPLE_TRACE)
endif()
