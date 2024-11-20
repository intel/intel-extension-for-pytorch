cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

set(TARGET esimd_kernels)

FILE(GLOB_RECURSE kernels_src *.cpp)
add_library(${TARGET} SHARED ${kernels_src})
set_target_properties(${TARGET} PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_compile_options(${TARGET} PRIVATE -fsycl -ffast-math)
if (USE_AOT_DEVLIST)
  target_compile_options(${TARGET} PRIVATE -fsycl-targets=spir64_gen)
endif()


# link openblas
set(LINKE_OPTS ${LINKE_OPTS}
  -fsycl
  -fsycl-device-code-split=per_kernel
)

if (USE_AOT_DEVLIST)
  set(LINKE_OPTS ${LINKE_OPTS} -fsycl-targets=spir64_gen)
endif()

set(PERF_LINK_OPTS "-doubleGRF")
set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -vc-disable-indvars-opt")
set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -vc-codegen")
set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -Xfinalizer -printregusage")
set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -Xfinalizer -enableBCR")
set(PERF_LINK_OPTS "${PERF_LINK_OPTS} -Xfinalizer -DPASTokenReduction")

set(LINKE_OPTS ${LINKE_OPTS} -Xs )

if (USE_AOT_DEVLIST)
  set(LINKE_OPTS ${LINKE_OPTS} "-device ${USE_AOT_DEVLIST} -options '${PERF_LINK_OPTS}'")
else()
  set(LINKE_OPTS ${LINKE_OPTS} "${PERF_LINK_OPTS}")
endif()
target_link_options(${TARGET} PRIVATE ${LINKE_OPTS})
target_link_libraries(${TARGET} PUBLIC ${GPU_TORCH_LIBS})



