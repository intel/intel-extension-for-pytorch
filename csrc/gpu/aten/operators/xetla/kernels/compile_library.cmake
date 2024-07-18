
function(add_library_with_options TARGET IS_DOUBLE_GRF)
  # link openblas
  set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS}
    -fsycl
    -fsycl-device-code-split=per_kernel
    -fsycl-max-parallel-link-jobs=${SYCL_MAX_PARALLEL_LINK_JOBS}
  )

  if (USE_AOT_DEVLIST)
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

  if (USE_AOT_DEVLIST)
    # Disable implementations for architectures not in USE_AOT_DEVLIST
    function (disable_architecture_by_aot ARCH REGULAR_EXPRESSION)
      string(REGEX MATCHALL ${REGULAR_EXPRESSION} USE_AOT_DEVLIST_${ARCH} "${USE_AOT_DEVLIST}")
      string(REPLACE ";" "," USE_AOT_DEVLIST_${ARCH} "${USE_AOT_DEVLIST_${ARCH}}")
      set(USE_AOT_DEVLIST_${ARCH} "${USE_AOT_DEVLIST_${ARCH}}" PARENT_SCOPE)
      message(STATUS "XeTLA: USE_AOT_DEVLIST_${ARCH}: ${USE_AOT_DEVLIST_${ARCH}}")
      if("${USE_AOT_DEVLIST_${ARCH}}" STRLESS_EQUAL "")
        set(USE_XETLA_${ARCH} OFF PARENT_SCOPE)
      endif()
    endfunction()
    disable_architecture_by_aot(XE_HPC "(pvc|xe-hpc)")
    disable_architecture_by_aot(XE_HPG "(ats-m150|acm-g10|acm-g11|acm-g12|xe-hpg)")
    disable_architecture_by_aot(XE_LPG "(mtl-m|mtl-s|mtl-p|xe-lpg|0x7d55|0x7dd5|0x7d57|0x7dd7)")

    set(XETLA_USE_AOT_DEVLIST "${USE_AOT_DEVLIST}")
    if (USE_XETLA_XE_HPC)  # Temporary fix as AOT fails of try to compile XE_HPC kernels for ats-m150 etc
      message(STATUS "XeTLA: XE_HPC suppress other aot targets")
      set(XETLA_USE_AOT_DEVLIST "${USE_AOT_DEVLIST_XE_HPC}")
    elseif(USE_XETLA_XE_HPG) # Temporary fix as AOT fails of try to compile XE_HPG kernels for mtl-p etc
      message(STATUS "XeTLA: XE_HPG suppress other aot targets")
      set(XETLA_USE_AOT_DEVLIST "${USE_AOT_DEVLIST_XE_HPG}")
    endif()
    message(STATUS "XeTLA: XETLA_USE_AOT_DEVLIST: ${XETLA_USE_AOT_DEVLIST}")
    set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "-device ${XETLA_USE_AOT_DEVLIST} -options '${XETLA_OFFLINE_OPTIONS}'")
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
  if (USE_AOT_DEVLIST)
    target_compile_options(${TARGET} PRIVATE -fsycl-targets=spir64_gen)
  endif()

endfunction()
