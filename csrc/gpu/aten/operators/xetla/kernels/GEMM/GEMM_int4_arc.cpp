#include "gemm_int4_arc.h"

namespace xpu {
namespace xetla {


#define HGEMM_WINT4_ARC_IMPL_FUNC(                                             \
      WG_M, WG_N, SG_M, SG_N, SG_K, DEQUANT_S, SLM_KS)                         \
  template void hgemm_wint4_arc<                                               \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);                                                       \
  template void hgemm_bias_wint4_arc<                                          \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const sycl::half* bias,                                                  \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);                                                       \
  template void hgemm_qkv_wint4_arc<                                           \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);                                                       \
  template void hgemm_qkv_bias_wint4_arc<                                      \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const sycl::half* bias,                                                  \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);                                                       \
  template void hgemm_res_wint4_arc<                                           \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const sycl::half* res,                                                   \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);                                                       \
  template void hgemm_silu_mul_wint4_arc<                                      \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const sycl::half* mul,                                                   \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);                                                       \
  template void hgemm_bias_silu_mul_wint4_arc<                                 \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const sycl::half* bias,                                                  \
      const sycl::half* mul,                                                   \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);                                                       \
  template void hgemm_bias_add_wint4_arc<                                      \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const sycl::half* bias,                                                  \
      const sycl::half* res,                                                   \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);                                                       \
  template void hgemm_silu_wint4_arc<                                          \
      sycl::half,                                                              \
      WG_M,                                                                    \
      WG_N,                                                                    \
      SG_M,                                                                    \
      SG_N,                                                                    \
      SG_K,                                                                    \
      DEQUANT_S,                                                               \
      SLM_KS,                                                                  \
      1,                                                                       \
      8,                                                                       \
      3>(                                                                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const uint8_t* b,                                                        \
      const uint8_t* b_zp,                                                     \
      const sycl::half* b_scale,                                               \
      const uint32_t m,                                                        \
      const uint32_t n,                                                        \
      const uint32_t k);


// k group ARC
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 16, 8);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 32, 8);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 64, 8);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 128, 8);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 256, 8);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 512, 8);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 1024, 8);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 16, 4);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 32, 4);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 64, 4);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 128, 4);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 256, 4);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 512, 4);
HGEMM_WINT4_ARC_IMPL_FUNC(8, 64, 8, 16, 16, 1024, 4);
} // namespace xetla
} // namespace xpu
