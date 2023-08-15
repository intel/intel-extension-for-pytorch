#pragma once

#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {

#define HGEMM_WINT4_DESC_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, DEQUANT_S, SLM_KS)                         \
  void                                                                                                 \
      hgemm_wint4_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##DEQUANT_S##_##SLM_KS##_(              \
          sycl::queue& queue,                                                                          \
          sycl::half* out,                                                                             \
          const sycl::half* a,                                                                         \
          const uint8_t* b,                                                                            \
          const uint8_t* b_zp,                                                                         \
          const sycl::half* b_scale,                                                                   \
          const int m,                                                                                 \
          const int n,                                                                                 \
          const int k);                                                                                \
  void                                                                                                 \
      hgemm_bias_wint4_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##DEQUANT_S##_##SLM_KS##_(         \
          sycl::queue& queue,                                                                          \
          sycl::half* out,                                                                             \
          const sycl::half* a,                                                                         \
          const uint8_t* b,                                                                            \
          const uint8_t* b_zp,                                                                         \
          const sycl::half* b_scale,                                                                   \
          const sycl::half* bias,                                                                      \
          const int m,                                                                                 \
          const int n,                                                                                 \
          const int k);                                                                                \
  void                                                                                                 \
      hgemm_bias_gelu_wint4_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##DEQUANT_S##_##SLM_KS##_(    \
          sycl::queue& queue,                                                                          \
          sycl::half* out,                                                                             \
          const sycl::half* a,                                                                         \
          const uint8_t* b,                                                                            \
          const uint8_t* b_zp,                                                                         \
          const sycl::half* b_scale,                                                                   \
          const sycl::half* bias,                                                                      \
          const int m,                                                                                 \
          const int n,                                                                                 \
          const int k);                                                                                \
  void                                                                                                 \
      hgemm_res_wint4_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##DEQUANT_S##_##SLM_KS##_(          \
          sycl::queue& queue,                                                                          \
          sycl::half* out,                                                                             \
          const sycl::half* a,                                                                         \
          const uint8_t* b,                                                                            \
          const uint8_t* b_zp,                                                                         \
          const sycl::half* b_scale,                                                                   \
          const sycl::half* res,                                                                       \
          const int m,                                                                                 \
          const int n,                                                                                 \
          const int k);                                                                                \
  void                                                                                                 \
      hgemm_bias_res_res_wint4_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##DEQUANT_S##_##SLM_KS##_( \
          sycl::queue& queue,                                                                          \
          sycl::half* out,                                                                             \
          const sycl::half* a,                                                                         \
          const uint8_t* b,                                                                            \
          const uint8_t* b_zp,                                                                         \
          const sycl::half* b_scale,                                                                   \
          const sycl::half* bias,                                                                      \
          const sycl::half* res0,                                                                      \
          const sycl::half* res1,                                                                      \
          const int m,                                                                                 \
          const int n,                                                                                 \
          const int k);                                                                                \
  //void                                                                                                 \
  //    hgemm_qkv_wint4_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##DEQUANT_S##_##SLM_KS##_(          \
  //        sycl::queue& queue,                                                                          \
  //        sycl::half* out0,                                                                            \
  //        sycl::half* out1,                                                                            \
  //        sycl::half* out2,                                                                            \
  //        const sycl::half* a,                                                                         \
  //        const uint8_t* b,                                                                            \
  //        const uint8_t* b_zp,                                                                         \
  //        const sycl::half* b_scale,                                                                   \
  //        const int m,                                                                                 \
  //        const int n,                                                                                 \
  //        const int k);                                                                                \
  //void                                                                                                 \
  //    hgemm_qkv_bias_wint4_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##DEQUANT_S##_##SLM_KS##_(     \
  //        sycl::queue& queue,                                                                          \
  //        sycl::half* out0,                                                                            \
  //        sycl::half* out1,                                                                            \
  //        sycl::half* out2,                                                                            \
  //        const sycl::half* a,                                                                         \
  //        const uint8_t* b,                                                                            \
  //        const uint8_t* b_zp,                                                                         \
  //        const sycl::half* b_scale,                                                                   \
  //        const sycl::half* bias,                                                                      \
  //        const int m,                                                                                 \
  //        const int n,                                                                                 \
  //        const int k);

// k group

//HGEMM_WINT4_DESC_FUNC(8, 256, 8, 16, 32, 128, 2);
//HGEMM_WINT4_DESC_FUNC(8, 64, 8, 16, 64, 128, 8);
//HGEMM_WINT4_DESC_FUNC(8, 512, 8, 16, 32, 128, 1);
//HGEMM_WINT4_DESC_FUNC(16, 256, 16, 16, 32, 128, 2);
//HGEMM_WINT4_DESC_FUNC(16, 64, 16, 16, 32, 128, 8);
//HGEMM_WINT4_DESC_FUNC(16, 512, 16, 16, 32, 128, 1);
//HGEMM_WINT4_DESC_FUNC(32, 256, 32, 16, 32, 128, 2);
//HGEMM_WINT4_DESC_FUNC(32, 64, 32, 16, 32, 128, 8);
//HGEMM_WINT4_DESC_FUNC(32, 128, 32, 16, 32, 128, 4);
//HGEMM_WINT4_DESC_FUNC(32, 512, 32, 16, 32, 128, 1);
//HGEMM_WINT4_DESC_FUNC(64, 256, 64, 16, 32, 128, 2);
//HGEMM_WINT4_DESC_FUNC(64, 128, 64, 16, 32, 128, 4);
//HGEMM_WINT4_DESC_FUNC(64, 512, 64, 16, 32, 128, 1);
//HGEMM_WINT4_DESC_FUNC(128, 256, 64, 16, 32, 128, 1);
//HGEMM_WINT4_DESC_FUNC(128, 512, 64, 32, 32, 128, 1);
//HGEMM_WINT4_DESC_FUNC(256, 256, 64, 32, 32, 128, 1);

// per channel
HGEMM_WINT4_DESC_FUNC(8, 256, 8, 16, 32, 0, 2);
HGEMM_WINT4_DESC_FUNC(8, 64, 8, 16, 64, 0, 8);
HGEMM_WINT4_DESC_FUNC(8, 512, 8, 16, 32, 0, 1);
HGEMM_WINT4_DESC_FUNC(16, 256, 16, 16, 32, 0, 2);
HGEMM_WINT4_DESC_FUNC(16, 64, 16, 16, 32, 0, 8);
HGEMM_WINT4_DESC_FUNC(16, 512, 16, 16, 32, 0, 1);
HGEMM_WINT4_DESC_FUNC(32, 256, 32, 16, 32, 0, 2);
HGEMM_WINT4_DESC_FUNC(32, 64, 32, 16, 32, 0, 8);
HGEMM_WINT4_DESC_FUNC(32, 128, 32, 16, 32, 0, 4);
HGEMM_WINT4_DESC_FUNC(32, 512, 32, 16, 32, 0, 1);
HGEMM_WINT4_DESC_FUNC(64, 256, 64, 16, 32, 0, 2);
HGEMM_WINT4_DESC_FUNC(64, 128, 64, 16, 32, 0, 4);
HGEMM_WINT4_DESC_FUNC(64, 512, 64, 16, 32, 0, 1);
HGEMM_WINT4_DESC_FUNC(128, 256, 64, 16, 32, 0, 1);
HGEMM_WINT4_DESC_FUNC(128, 512, 64, 32, 32, 0, 1);
HGEMM_WINT4_DESC_FUNC(256, 256, 64, 32, 32, 0, 1);
} // namespace xetla
} // namespace xpu
