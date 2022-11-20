# ---[ Build flags
# TODO: CPU: include(${IPEX_ROOT_DIR}/cmake/python/BuildOptions.cmake)

FILE(GLOB IPEX_PYTHON_CPP_CPU_SRCS *.cpp)
add_library(PY_CPU_OBJ OBJECT ${IPEX_PYTHON_CPP_CPU_SRCS})

# includes
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_CPU_ROOT_DIR}/aten)
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_CPU_ROOT_DIR}/utils)

target_include_directories(PY_CPU_OBJ PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(PY_CPU_OBJ PUBLIC ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include/)

if(PYBIND11_CL_FLAGS)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND11_CL_FLAGS}")
else()
  message(FATAL_ERROR, "Cannot get pybind11 abi compiler flags")
endif()

# Set Python include dir
if(PYTHON_INCLUDE_DIR)
  target_include_directories(PY_CPU_OBJ PUBLIC ${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR, "Cannot find installed Python head file directory")
endif()
