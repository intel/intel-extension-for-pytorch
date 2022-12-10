# ---[ Build flags
include(${IPEX_ROOT_DIR}/cmake/cpu/BuildFlags.cmake)

FILE(GLOB IPEX_PYTHON_CPP_CPU_SRCS *.cpp)
add_library(PY_CPU_OBJ OBJECT ${IPEX_PYTHON_CPP_CPU_SRCS})

# includes
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_CPU_ROOT_DIR})
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_CPU_ROOT_DIR}/aten)
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_CPU_ROOT_DIR}/aten/utils)
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_CPU_ROOT_DIR}/dyndisp)
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_CPU_ROOT_DIR}/runtime)
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_ROOT_DIR}/csrc)
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_ROOT_DIR}/csrc/jit/codegen/onednn)
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_ROOT_DIR}/csrc/utils)
target_include_directories(PY_CPU_OBJ PUBLIC ${IPEX_CPU_ROOT_DIR}/utils)

target_include_directories(PY_CPU_OBJ PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(PY_CPU_OBJ PUBLIC ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include/)

target_include_directories(PY_CPU_OBJ PUBLIC ${PYTHON_INCLUDE_DIR})
target_include_directories(PY_CPU_OBJ PUBLIC ${pybind11_INCLUDE_DIRS})
