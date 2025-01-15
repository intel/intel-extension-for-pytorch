FILE(GLOB aten_ops_src "operators/*.cpp" "operators/sparse/*.cpp" "operators/transformers/*.cpp" "operators/fp8/*.cpp" "operators/transformers_esimd/*.cpp" "operators/torch-xpu-ops/*.cpp" "operators/torch-xpu-ops/sycl/*.cpp")

# Reset the order of src files to compile slow files first
list(APPEND slow_patterns
  Pow
  MathCompare
  Activation
  UnaryOps
  BinaryOps
  TopK
  Sort
  )

set(slow_srcs)
foreach (fn_pattern ${slow_patterns})
  foreach (sfile ${aten_ops_src})
    string(REGEX MATCHALL "${fn_pattern}[_a-zA-Z0-9]*\.cpp$" FN_LIST ${sfile})
    if (FN_LIST)
      list(APPEND slow_srcs ${sfile})
    endif()
  endforeach()
endforeach()

set(fast_srcs)
foreach(sfile ${aten_ops_src})
  if (NOT ${sfile} IN_LIST slow_srcs)
    list(APPEND fast_srcs ${sfile})
  endif()
endforeach()

set(IPEX_ATEN_OPS_SRCS ${slow_srcs} ${fast_srcs} PARENT_SCOPE)

FILE(GLOB aten_core_src "core/*.cpp" "core/detail/*.cpp")
FILE(GLOB aten_onednn_src "../oneDNN/*.cpp")
FILE(GLOB aten_quant_src "quantized/*.cpp")
FILE(GLOB aten_tensor_src "tensor/*.cpp")

set(IPEX_ATEN_CORE_SRCS
  ${aten_core_src}
  ${aten_onednn_src}
  ${aten_quant_src}
  ${aten_tensor_src}
  PARENT_SCOPE)

# OPs that use XeTLA kernels when USE_XETLA
FILE(GLOB xetla_ops_src
  "operators/transformers/*.cpp"
  "operators/Blas.cpp"
  "operators/GRU.cpp"
  "operators/XEGEMM_INT4.cpp"
  "operators/XeGemm.cpp"
  "operators/XeInt4Dequantize.cpp"
  "operators/Punica.cpp"
)
set(IPEX_ATEN_XETLA_OPS_SRCS "${xetla_ops_src}" PARENT_SCOPE)
set(esimd_ops_src "operators/transformers_esimd/attention_mtl.cpp")
set(IPEX_ATEN_ESIMD_OPS_SRCS "${esimd_ops_src}" PARENT_SCOPE)
set(XETLA_LIBS ${XETLA_LIBS} PARENT_SCOPE)
