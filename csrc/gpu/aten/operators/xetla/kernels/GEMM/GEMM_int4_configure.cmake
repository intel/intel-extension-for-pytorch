function(GEMM_int4_configure XETLA_USED_ARCHS)
set(GEMM_INT4_LIBS)

# set(L_QUANT_MODE "I4_SYM" "I4_ASYM_ZERO_NO_DEGRAD")
set(L_QUANT_MODE "I4_SYM")

# avoid too long file name
set(QUANT_MODE_FLAG_I4_SYM "qmode1")
set(QUANT_MODE_FLAG_I4_ASYM_ZERO_NO_DEGRAD "qmode2")

# cmake arch to gpu_arch::xxxx
set(gpu_arch_xe_lpg "XeLpg")
set(gpu_arch_xe_hpg "XeHpg")
set(gpu_arch_xe_hpc "XeHpc")


function(configure_ SCALAR_TS ARCH QUANT_MODE WG_M WG_N SG_M SG_N SG_K SLM_KS L3_KS SYNC_FREQ STAGES)
    set(SUFFIX0 "${WG_M}_${WG_N}_${SG_N}_${SG_K}_${SLM_KS}_${L3_KS}_${SYNC_FREQ}_${STAGES}")
    foreach(SCALAR_T ${SCALAR_TS})
        set(SUFFIX "${ARCH}_${QUANT_MODE_FLAG_${QUANT_MODE}}_${SCALAR_T}_${SUFFIX0}")
        configure_file(GEMM/GEMM_int4.cpp.in "GEMM_int4_${SUFFIX}.cpp")
        if(WG_M GREATER 32)
            set(IS_GEMM TRUE)
        else()
            set(IS_GEMM FALSE)
        endif()
        add_library_with_options("xetla_${SUFFIX}" ${IS_GEMM} "${CMAKE_CURRENT_BINARY_DIR}/GEMM_int4_${SUFFIX}.cpp")
        list(APPEND GEMM_INT4_LIBS "xetla_${SUFFIX}")
        set(GEMM_INT4_LIBS ${GEMM_INT4_LIBS} PARENT_SCOPE)
    endforeach()
endfunction()

foreach(ARCH ${XETLA_USED_ARCHS})
foreach(QUANT_MODE ${L_QUANT_MODE})
    set(arch_tag "${gpu_arch_${ARCH}}")

    if(ARCH STREQUAL "xe_lpg")
        if(QUANT_MODE STREQUAL "I4_ASYM_ZERO_NO_DEGRAD")
            continue()
        endif()
        configure_("fp16"      ${arch_tag} ${QUANT_MODE} 1   1   1  1  256  1 1 0 0)
        configure_("fp16"      ${arch_tag} ${QUANT_MODE} 1   1   1  1  512  1 1 0 0)
        configure_("fp16"      ${arch_tag} ${QUANT_MODE} 4   1   4  1  128  1 1 0 0)

    elseif(ARCH STREQUAL "xe_hpg")
        configure_("fp16;bf16"      ${arch_tag} ${QUANT_MODE} 1   1   1  1  256  1 1 0 0)
        configure_("fp16;bf16"      ${arch_tag} ${QUANT_MODE} 1   1   1  1  512  1 1 0 0)

    elseif(ARCH STREQUAL "xe_hpc")
        configure_("fp16;bf16"      ${arch_tag} ${QUANT_MODE} 1   1   1  1  256  1 1 1 3)
        configure_("fp16;bf16"      ${arch_tag} ${QUANT_MODE} 1   1   1  1  512  1 1 1 3)

    else()
        message(FATAL_ERROR "Unsupported arch: ${ARCH}")
    endif()
endforeach()
endforeach()

list(LENGTH GEMM_INT4_LIBS GEMM_INT4_LIBS_LENGTH)
set(GEMM_INT4_LIBS ${GEMM_INT4_LIBS} PARENT_SCOPE)
message(STATUS "Generated GEMM_int4 sources: ${GEMM_INT4_LIBS_LENGTH}")
endfunction()
