
function(add_library_with_options TARGET IS_DOUBLE_GRF AOT_DEVLIST)
  # link openblas
  set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS}
    -fsycl
    -fsycl-device-code-split=per_kernel
    -fsycl-max-parallel-link-jobs=${SYCL_MAX_PARALLEL_LINK_JOBS}
  )

  if (AOT_DEVLIST)
    set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -fsycl-targets=spir64_gen)
  endif()

  # set RPATH
  if(NOT WINDOWS)
    foreach(RPATH ${RPATHS_LIST})
      set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} ${RPATH})
    endforeach()
    set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "-Wl,--disable-new-dtags")
  endif()

  set(XETLA_OFFLINE_OPTIONS "")
  if(${IS_DOUBLE_GRF})
    set(XETLA_OFFLINE_OPTIONS "-doubleGRF")
  endif()
  set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -vc-disable-indvars-opt")
  set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -vc-codegen")
  # For registers usage verbose at AOT
  set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -printregusage")
  # Enable bank conflict reduction.
  set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -enableBCR")
  # Optimization to reduce the tokens used for DPAS instruction.
  set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -DPASTokenReduction")

  set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -Xs )

  if (AOT_DEVLIST)
    set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "-device ${AOT_DEVLIST} -options '${XETLA_OFFLINE_OPTIONS}'")
  else()
    set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "${XETLA_OFFLINE_OPTIONS}")
  endif()

  add_library(${TARGET} SHARED ${ARGN})

  target_include_directories(${TARGET} PRIVATE .)
  target_include_directories(${TARGET} PUBLIC ${XETLA_INCLUDE_DIR})
  target_include_directories(${TARGET} PUBLIC ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${TARGET} PUBLIC ${GPU_TORCH_LIBS})

  # Set visibility to hidden to close the differences of Windows & Linux
  set_target_properties(${TARGET} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  if(!${IS_DOUBLE_GRF})
    target_compile_definitions(${TARGET} PRIVATE NORMAL_GRF)
  endif()
  target_compile_definitions(${TARGET} PRIVATE BUILD_XETLA_KERNEL_LIB)

  # Feature flag macros should be public for uses in its dependent libraries
  target_compile_definitions(${TARGET} PUBLIC USE_XETLA)
  foreach(available_arch IN LISTS XETLA_AVAILABLE_ARCHS)
    string(TOUPPER ${available_arch} ARCH)
    if(USE_XETLA_${ARCH})
      target_compile_definitions(${TARGET} PUBLIC "USE_XETLA_${ARCH}")
    endif()
  endforeach()

  target_link_options(${TARGET} PRIVATE ${XETLA_KERNEL_FLAGS})
  target_compile_options(${TARGET} PRIVATE -fsycl)
  if (AOT_DEVLIST)
    target_compile_options(${TARGET} PRIVATE -fsycl-targets=spir64_gen)
  endif()

endfunction()
