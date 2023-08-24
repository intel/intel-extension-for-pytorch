#pragma once

#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {

#define HGEMM_FUNC_NAME(                                     \
    HEAD, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HEAD##_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_

#define HGEMM_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(hgemm, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(hgemm_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_RES_RES_FUNC(                       \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(                                     \
      hgemm_bias_res_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_GELU_FUNC(                          \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(                                     \
      hgemm_bias_gelu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RESMUL_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(                                                           \
      hgemm_resmul, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_SILU_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(hgemm_silu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RES_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(hgemm_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_QKV_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(hgemm_qkv, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_QKV_BIAS_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(                                                             \
      hgemm_qkv_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_RES_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(                                                             \
      hgemm_bias_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

#define HGEMM_ENUMERATE_FUNC_DESCS(                                            \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)                         \
  void HGEMM_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(          \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_BIAS_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(     \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_BIAS_RES_RES_FUNC(                                                \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const sycl::half* res0,                                                  \
      const sycl::half* res1,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_BIAS_GELU_FUNC(                                                   \
      WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(                      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_RESMUL_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(   \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* mul,                                                   \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_SILU_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(     \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_RES_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(      \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* res,                                                   \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_QKV_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(      \
      sycl::queue & queue,                                                     \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_QKV_BIAS_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue,                                                     \
      sycl::half * out0,                                                       \
      sycl::half * out1,                                                       \
      sycl::half * out2,                                                       \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const int m,                                                             \
      const int n,                                                             \
      const int k);                                                            \
  void HGEMM_BIAS_RES_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const sycl::half* bias,                                                  \
      const sycl::half* res,                                                   \
      const sycl::half res_scale,                                              \
      const int m,                                                             \
      const int n,                                                             \
      const int k);

// clang-format off
#define HGEMM_COMMA ,
#define HGEMM_NUM_POLICIES 23
#define HGEMM_ENUMERATE_POLICIES(_, T) \
    _(8, 64, 8, 16, 32, 8, true)T      \
    _(8, 256, 8, 16, 16, 2, true)T     \
    _(8, 512, 8, 16, 16, 1, true)T     \
    _(16, 64, 16, 16, 16, 8, true)T    \
    _(16, 256, 16, 16, 16, 2, true)T   \
    _(16, 512, 16, 16, 16, 1, true)T   \
    _(32, 64, 32, 16, 16, 8, true)T    \
    _(32, 64, 8, 16, 16, 2, true)T    \
    _(32, 128, 32, 16, 16, 4, true)T   \
    _(32, 256, 32, 16, 16, 2, true)T   \
    _(32, 512, 32, 16, 16, 1, true)T   \
    _(64, 128, 64, 16, 16, 4, true)T   \
    _(64, 256, 64, 16, 16, 2, true)T   \
    _(64, 512, 64, 16, 16, 1, true)T   \
    _(128, 128, 32, 32, 32, 2, true)T  \
    _(128, 256, 64, 16, 16, 1, true)T  \
    _(128, 512, 64, 32, 16, 1, true)T  \
    _(256, 256, 64, 32, 16, 1, true)T  \
    _(256, 256, 32, 64, 16, 1, true)T  \
    _(256, 256, 32, 64, 32, 1, true)T  \
    _(128, 64, 16, 16, 64, 1, true)T   \
    _(128, 128, 16, 32, 64, 1, true)T  \
    _(128, 256, 32, 32, 16, 1, true)T  \
    _(8, 64, 8, 16, 32, 8, false)T     \
    _(8, 256, 8, 16, 16, 2, false)T    \
    _(8, 512, 8, 16, 16, 1, false)T    \
    _(16, 64, 16, 16, 16, 8, false)T   \
    _(16, 256, 16, 16, 16, 2, false)T  \
    _(16, 512, 16, 16, 16, 1, false)T  \
    _(32, 64, 32, 16, 16, 8, false)T   \
    _(32, 64, 8, 16, 16, 2, false)T    \
    _(32, 128, 32, 16, 16, 4, false)T  \
    _(32, 256, 32, 16, 16, 2, false)T  \
    _(32, 512, 32, 16, 16, 1, false)T  \
    _(64, 128, 64, 16, 16, 4, false)T  \
    _(64, 256, 64, 16, 16, 2, false)T  \
    _(64, 512, 64, 16, 16, 1, false)T  \
    _(128, 128, 32, 32, 32, 2, false)T \
    _(128, 256, 64, 16, 16, 1, false)T \
    _(128, 512, 64, 32, 16, 1, false)T \
    _(256, 256, 64, 32, 16, 1, false)T \
    _(256, 256, 32, 64, 16, 1, false)T  \
    _(256, 256, 32, 64, 32, 1, false)T  \
    _(128, 64, 16, 16, 64, 1, false)T   \
    _(128, 128, 16, 32, 64, 1, false)T \
    _(128, 256, 32, 32, 16, 1, false)T
// clang-format on

HGEMM_ENUMERATE_POLICIES(HGEMM_ENUMERATE_FUNC_DESCS, )

#define HGEMM_POLICY_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  "_" #WG_M "x" #WG_N "_" #SG_M "x" #SG_N "x" #SG_K "_" #SLM_KS              \
  "_" #B_ROW_MAJOR "_"
const char* hgemm_policy_names[2 * HGEMM_NUM_POLICIES] = {
    HGEMM_ENUMERATE_POLICIES(HGEMM_POLICY_NAME, HGEMM_COMMA)};

#define HGEMM_ENUMERATE_FUNC_TRAITS(                   \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  { WG_M, WG_N }
int hgemm_policies_wg_mnk[2 * HGEMM_NUM_POLICIES][2]{
    HGEMM_ENUMERATE_POLICIES(HGEMM_ENUMERATE_FUNC_TRAITS, HGEMM_COMMA)};

void (*hgemm_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_FUNC, HGEMM_COMMA)};

void (*hgemm_bias_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_BIAS_FUNC, HGEMM_COMMA)};

void (*hgemm_bias_res_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {
    HGEMM_ENUMERATE_POLICIES(HGEMM_BIAS_RES_RES_FUNC, HGEMM_COMMA)};

void (*hgemm_bias_gelu_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_BIAS_GELU_FUNC, HGEMM_COMMA)};

void (*hgemm_resmul_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_RESMUL_FUNC, HGEMM_COMMA)};

void (*hgemm_silu_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_SILU_FUNC, HGEMM_COMMA)};

void (*hgemm_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_RES_FUNC, HGEMM_COMMA)};

void (*hgemm_qkv_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_QKV_FUNC, HGEMM_COMMA)};

void (*hgemm_qkv_bias_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_QKV_BIAS_FUNC, HGEMM_COMMA)};

void (*hgemm_bias_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES(HGEMM_BIAS_RES_FUNC, HGEMM_COMMA)};

struct gemm_cfg_meta {
  float wg_eff;
  float num_ss;
  float aspect_r;
  int idx;
};

inline int select_gemm_special_config(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major) {
  if (m <= 8 && n <= 4096) {
    return is_b_row_major ? 7 : 7 + HGEMM_NUM_POLICIES;
  } else if (m <= 16) {
    return -1;
  } else if (m <= 256 && n >= 16384) {
    return is_b_row_major ? 18 : 18 + HGEMM_NUM_POLICIES;
  } else if (((m + 255) / 256) * ((n + 255) / 256) >= 64) {
    return is_b_row_major ? 19 : 19 + HGEMM_NUM_POLICIES;
  } else if (m <= 128 && n == 4096) {
    return is_b_row_major ? 20 : 20 + HGEMM_NUM_POLICIES;
  } else if (m <= 256 && n == 4096) {
    return is_b_row_major ? 21 : 21 + HGEMM_NUM_POLICIES;
  } else if (m <= 512 && n == 4096) {
    return is_b_row_major ? 22 : 22 + HGEMM_NUM_POLICIES;
  }
  return -1;
}

inline int select_gemm_config(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major,
    const int TOTAL_SS = 64) {
  int idx = select_gemm_special_config(m, n, k, is_b_row_major);
  if (idx >= 0)
    return idx;
  std::vector<gemm_cfg_meta> metas;
  for (int i = 0; i < HGEMM_NUM_POLICIES; i++) {
    gemm_cfg_meta meta;
    int wg_m = hgemm_policies_wg_mnk[i][0];
    int wg_n = hgemm_policies_wg_mnk[i][1];
    int ms = (m + wg_m - 1) / wg_m;
    int ns = (n + wg_n - 1) / wg_n;
    meta.num_ss = ms * ns;
    int vm = m > wg_m ? wg_m : m;
    int vn = n > wg_n ? wg_n : n;
    meta.wg_eff = (float)vm * vn / (float)wg_m / (float)wg_n;
    meta.idx = i;
    meta.aspect_r = std::max((float)wg_m / wg_n, (float)wg_n / wg_m);
    metas.push_back(meta);
  }
  std::sort(metas.begin(), metas.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.wg_eff != rhs.wg_eff)
      return lhs.wg_eff > rhs.wg_eff;
    else if (lhs.num_ss != rhs.num_ss)
      return lhs.num_ss < rhs.num_ss;
    else
      return lhs.aspect_r < rhs.aspect_r;
  });
  idx = metas[0].idx;
  return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
}

} // namespace xetla
} // namespace xpu
