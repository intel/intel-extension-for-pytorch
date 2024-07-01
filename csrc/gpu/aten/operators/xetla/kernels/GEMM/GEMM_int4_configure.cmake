function(GEMM_int4_configure XETLA_USED_ARCHS)
set(GEMM_INT4_A_SRCS)
set(GEMM_INT4_I_SRCS)

set(L_QUANT_MODE "S4_FULLRANGE_NO_ZP" "S4_ASYM_ZERO_NO_DEGRAD")

# aviod too long file name
set(QUANT_MODE_FLAG_S4_FULLRANGE_NO_ZP "qmode1")
set(QUANT_MODE_FLAG_S4_ASYM_ZERO_NO_DEGRAD "qmode2")

# cmake arch to gpu_arch::xxxx
set(gpu_arch_xe_lpg "XeLpg")
set(gpu_arch_xe_hpg "XeHpg")
set(gpu_arch_xe_hpc "XeHpc")


function(configure_ SCALAR_TS ARCH QUANT_MODE WG_M WG_N SG_M SG_N SG_K SLM_KS L3_KS SYNC_FREQ STAGES)
    set(SUFFIX "${WG_M}_${WG_N}_${SG_N}_${SG_K}_${SLM_KS}_${L3_KS}_${SYNC_FREQ}_${STAGES}")

    set(SUFFIX_I "${ARCH}_${QUANT_MODE_FLAG_${QUANT_MODE}}_${SUFFIX}")
    configure_file(GEMM/GEMM_int4_impl_i.cpp.in "GEMM_int4_impl_i_${SUFFIX_I}.cpp")
    list(APPEND GEMM_INT4_I_SRCS "${CMAKE_CURRENT_BINARY_DIR}/GEMM_int4_impl_i_${SUFFIX_I}.cpp")
    set(GEMM_INT4_I_SRCS ${GEMM_INT4_I_SRCS} PARENT_SCOPE)

    foreach(SCALAR_T ${SCALAR_TS})
        set(SUFFIX_A "${ARCH}_${QUANT_MODE_FLAG_${QUANT_MODE}}_${SCALAR_T}_${SUFFIX}")
        configure_file(GEMM/GEMM_int4_impl_a.cpp.in "GEMM_int4_impl_a_${SUFFIX_A}.cpp")
        list(APPEND GEMM_INT4_A_SRCS "${CMAKE_CURRENT_BINARY_DIR}/GEMM_int4_impl_a_${SUFFIX_A}.cpp")
        set(GEMM_INT4_A_SRCS ${GEMM_INT4_A_SRCS} PARENT_SCOPE)
    endforeach()
endfunction()

foreach(ARCH ${XETLA_USED_ARCHS})
foreach(QUANT_MODE ${L_QUANT_MODE})
    set(arch_tag "${gpu_arch_${ARCH}}")

    if(ARCH STREQUAL "xe_lpg")
        if(QUANT_MODE STREQUAL "S4_ASYM_ZERO_NO_DEGRAD")
            continue()
        endif()
        configure_("fp16"      ${arch_tag} ${QUANT_MODE} 1   128 1  16 16 8 1 0 1)
        configure_("fp16"      ${arch_tag} ${QUANT_MODE} 16  32  8  16 16 8 1 0 1)
        
    elseif(ARCH STREQUAL "xe_hpg")
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 8   64  8  16 16 8 1 0 0)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 8   64  8  16 16 4 1 0 0)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 32  256 16 16 32 1 1 0 0)

    elseif(ARCH STREQUAL "xe_hpc")
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 8   256 8  16 32 2 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 8   64  8  16 64 8 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 8   512 8  16 32 1 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 16  256 16 16 32 2 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 16  64  16 16 32 8 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 16  512 16 16 32 1 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 32  256 32 16 32 2 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 32  64  32 16 32 8 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 32  128 32 16 32 4 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 32  512 32 16 32 1 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 64  256 64 16 32 2 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 64  128 64 16 32 4 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 64  512 64 16 32 1 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 128 256 64 16 32 1 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 128 512 64 32 32 1 1 1 3)
        configure_("fp16;bf16" ${arch_tag} ${QUANT_MODE} 256 256 64 32 32 1 1 1 3)

    else()
        message(FATAL_ERROR "Unsupported arch: ${ARCH}")
    endif()
endforeach()
endforeach()

list(LENGTH GEMM_INT4_A_SRCS GEMM_INT4_A_SRCS_LENGTH)
list(LENGTH GEMM_INT4_I_SRCS GEMM_INT4_I_SRCS_LENGTH)
message(STATUS "Generated GEMM_int4_impl_a sources: ${GEMM_INT4_A_SRCS_LENGTH}")
message(STATUS "Generated GEMM_int4_impl_i sources: ${GEMM_INT4_I_SRCS_LENGTH}")
set(GEMM_INT4_SRCS ${GEMM_INT4_A_SRCS} ${GEMM_INT4_I_SRCS} PARENT_SCOPE)
endfunction()
