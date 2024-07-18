add_subdirectory(${IPEX_GPU_ROOT_DIR}/aten/operators/xetla/kernels)

set(XETLA_LIBS ${XETLA_LIBS} PARENT_SCOPE)
