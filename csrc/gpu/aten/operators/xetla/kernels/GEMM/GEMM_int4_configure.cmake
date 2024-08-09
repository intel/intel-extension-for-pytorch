function(GEMM_int4_configure XETLA_USED_ARCHS)
set(GEMM_INT4_LIBS)
# set(GEMM_INT4_I_SRCS)

# set(L_QUANT_MODE "S4_FULLRANGE_NO_ZP" "S4_ASYM_ZERO_NO_DEGRAD")
set(L_QUANT_MODE "S4_FULLRANGE_NO_ZP")

# aviod too long file name
set(QUANT_MODE_FLAG_S4_FULLRANGE_NO_ZP "qmode1")
set(QUANT_MODE_FLAG_S4_ASYM_ZERO_NO_DEGRAD "qmode2")

# cmake arch to gpu_arch::xxxx
set(gpu_arch_xe_lpg "XeLpg")
set(gpu_arch_xe_hpg "XeHpg")
set(gpu_arch_xe_hpc "XeHpc")


function(configure_ SCALAR_TS ARCH AOT_DEVLIST QUANT_MODE WG_M WG_N SG_M SG_N SG_K SLM_KS L3_KS SYNC_FREQ STAGES)
    set(SUFFIX "${WG_M}_${WG_N}_${SG_N}_${SG_K}_${SLM_KS}_${L3_KS}_${SYNC_FREQ}_${STAGES}")

    # set(SUFFIX_I "${ARCH}_${QUANT_MODE_FLAG_${QUANT_MODE}}_${SUFFIX}")
    # configure_file(GEMM/GEMM_int4_impl_i.cpp.in "GEMM_int4_impl_i_${SUFFIX_I}.cpp")
    # list(APPEND GEMM_INT4_I_SRCS "${CMAKE_CURRENT_BINARY_DIR}/GEMM_int4_impl_i_${SUFFIX_I}.cpp")
    # set(GEMM_INT4_I_SRCS ${GEMM_INT4_I_SRCS} PARENT_SCOPE)

    foreach(SCALAR_T ${SCALAR_TS})
        set(SUFFIX_A "${ARCH}_${QUANT_MODE_FLAG_${QUANT_MODE}}_${SCALAR_T}_${SUFFIX}")
        configure_file(GEMM/GEMM_int4_impl_a.cpp.in "GEMM_int4_impl_a_${SUFFIX_A}.cpp")
        if(WG_M GREATER 32)
            set(IS_GEMM TRUE)
        else()
            set(IS_GEMM FALSE)
        endif()
        add_library_with_options("xetla_${SUFFIX_A}" ${IS_GEMM} "${AOT_DEVLIST}" "${CMAKE_CURRENT_BINARY_DIR}/GEMM_int4_impl_a_${SUFFIX_A}.cpp")
        list(APPEND GEMM_INT4_LIBS "xetla_${SUFFIX_A}")
        set(GEMM_INT4_LIBS ${GEMM_INT4_LIBS} PARENT_SCOPE)
    endforeach()
endfunction()

foreach(ARCH ${XETLA_USED_ARCHS})
foreach(QUANT_MODE ${L_QUANT_MODE})
    string(TOUPPER ${ARCH} arch_upper)
    set(arch_tag "${gpu_arch_${ARCH}}")
    set(aot_tag "${XETLA_USE_AOT_DEVLIST_${arch_upper}}")

    if(ARCH STREQUAL "xe_lpg")
        if(QUANT_MODE STREQUAL "S4_ASYM_ZERO_NO_DEGRAD")
            continue()
        endif()
        configure_("fp16"      ${arch_tag} "${aot_tag}" ${QUANT_MODE} 1   1   1  1  256  1 1 0 0)
        configure_("fp16"      ${arch_tag} "${aot_tag}" ${QUANT_MODE} 1   1   1  1  512  1 1 0 0)
        configure_("fp16"      ${arch_tag} "${aot_tag}" ${QUANT_MODE} 4   1   4  1  128  1 1 0 0)

    elseif(ARCH STREQUAL "xe_hpg")
        configure_("fp16"      ${arch_tag} "${aot_tag}" ${QUANT_MODE} 1   1   1  1  256  1 1 0 0)
        configure_("fp16"      ${arch_tag} "${aot_tag}" ${QUANT_MODE} 1   1   1  1  512  1 1 0 0)

    elseif(ARCH STREQUAL "xe_hpc")
        configure_("fp16"      ${arch_tag} "${aot_tag}" ${QUANT_MODE} 1   1   1  1  256  1 1 0 0)
        configure_("fp16"      ${arch_tag} "${aot_tag}" ${QUANT_MODE} 1   1   1  1  512  1 1 0 0)

    else()
        message(FATAL_ERROR "Unsupported arch: ${ARCH}")
    endif()
endforeach()
endforeach()

list(LENGTH GEMM_INT4_LIBS GEMM_INT4_LIBS_LENGTH)
set(GEMM_INT4_LIBS ${GEMM_INT4_LIBS} PARENT_SCOPE)
# list(LENGTH GEMM_INT4_I_SRCS GEMM_INT4_I_SRCS_LENGTH)
message(STATUS "Generated GEMM_int4_impl_a sources: ${GEMM_INT4_LIBS_LENGTH}")
endfunction()
