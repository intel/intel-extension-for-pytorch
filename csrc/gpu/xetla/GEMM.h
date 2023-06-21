#pragma once

#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {

#define HGEMM_DESC_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS)                      \
  void hgemm_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_(             \
      sycl::queue& queue,                                                          \
      sycl::half* out,                                                             \
      const sycl::half* a,                                                         \
      const sycl::half* b,                                                         \
      const int m,                                                                 \
      const int n,                                                                 \
      const int k);                                                                \
  void hgemm_bias_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_(        \
      sycl::queue& queue,                                                          \
      sycl::half* out,                                                             \
      const sycl::half* a,                                                         \
      const sycl::half* b,                                                         \
      const sycl::half* bias,                                                      \
      const int m,                                                                 \
      const int n,                                                                 \
      const int k);                                                                \
  void                                                                             \
      hgemm_bias_res_res_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_( \
          sycl::queue& queue,                                                      \
          sycl::half* out,                                                         \
          const sycl::half* a,                                                     \
          const sycl::half* b,                                                     \
          const sycl::half* bias,                                                  \
          const sycl::half* res0,                                                  \
          const sycl::half* res1,                                                  \
          const int m,                                                             \
          const int n,                                                             \
          const int k);                                                            \
  void                                                                             \
      hgemm_bias_gelu_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_(    \
          sycl::queue& queue,                                                      \
          sycl::half* out,                                                         \
          const sycl::half* a,                                                     \
          const sycl::half* b,                                                     \
          const sycl::half* bias,                                                  \
          const int m,                                                             \
          const int n,                                                             \
          const int k);

HGEMM_DESC_FUNC(32, 64, 8, 16, 16, 2)
HGEMM_DESC_FUNC(8, 512, 8, 16, 16, 1)

void hgemm_qkv_8x128_8x16x32_4(
    sycl::queue& queue,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const int m,
    const int n,
    const int k);

} // namespace xetla
} // namespace xpu
