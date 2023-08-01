#pragma once

#include <sycl/sycl.hpp>
#include <algorithm>
#include <vector>

namespace xpu {
namespace xetla {

#define HGEMM_DESC_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)                         \
  void                                                                                             \
      hgemm_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(              \
          sycl::queue& queue,                                                                      \
          sycl::half* out,                                                                         \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_bias_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(         \
          sycl::queue& queue,                                                                      \
          sycl::half* out,                                                                         \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const sycl::half* bias,                                                                  \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_bias_res_res_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_( \
          sycl::queue& queue,                                                                      \
          sycl::half* out,                                                                         \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const sycl::half* bias,                                                                  \
          const sycl::half* res0,                                                                  \
          const sycl::half* res1,                                                                  \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_bias_gelu_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(    \
          sycl::queue& queue,                                                                      \
          sycl::half* out,                                                                         \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const sycl::half* bias,                                                                  \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_resmul_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(       \
          sycl::queue& queue,                                                                      \
          sycl::half* out,                                                                         \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const sycl::half* mul,                                                                   \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_silu_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(         \
          sycl::queue& queue,                                                                      \
          sycl::half* out,                                                                         \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_res_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(          \
          sycl::queue& queue,                                                                      \
          sycl::half* out,                                                                         \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const sycl::half* res,                                                                   \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_qkv_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(          \
          sycl::queue& queue,                                                                      \
          sycl::half* out0,                                                                        \
          sycl::half* out1,                                                                        \
          sycl::half* out2,                                                                        \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_qkv_bias_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(     \
          sycl::queue& queue,                                                                      \
          sycl::half* out0,                                                                        \
          sycl::half* out1,                                                                        \
          sycl::half* out2,                                                                        \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const sycl::half* bias,                                                                  \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);                                                                            \
  void                                                                                             \
      hgemm_bias_res_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(     \
          sycl::queue& queue,                                                                      \
          sycl::half* out,                                                                         \
          const sycl::half* a,                                                                     \
          const sycl::half* b,                                                                     \
          const sycl::half* bias,                                                                  \
          const sycl::half* res,                                                                   \
          const sycl::half res_scale,                                                              \
          const int m,                                                                             \
          const int n,                                                                             \
          const int k);

HGEMM_DESC_FUNC(32, 64, 8, 16, 16, 2, true)
HGEMM_DESC_FUNC(8, 512, 8, 16, 16, 1, true)
HGEMM_DESC_FUNC(16, 256, 8, 16, 16, 1, true)
HGEMM_DESC_FUNC(8, 128, 8, 16, 16, 4, true)
HGEMM_DESC_FUNC(32, 256, 8, 32, 16, 1, true)
HGEMM_DESC_FUNC(16, 128, 8, 16, 16, 1, true)
HGEMM_DESC_FUNC(8, 256, 8, 32, 16, 4, true)
HGEMM_DESC_FUNC(8, 512, 8, 32, 16, 2, true)
HGEMM_DESC_FUNC(256, 256, 32, 64, 32, 1, true)
HGEMM_DESC_FUNC(8, 128, 8, 16, 32, 4, true)
HGEMM_DESC_FUNC(32, 128, 8, 32, 32, 2, true)
HGEMM_DESC_FUNC(32, 64, 8, 16, 32, 2, true)
HGEMM_DESC_FUNC(256, 256, 32, 64, 16, 1, true)
HGEMM_DESC_FUNC(64, 128, 64, 16, 32, 4, true)

HGEMM_DESC_FUNC(32, 64, 8, 16, 16, 2, false)
HGEMM_DESC_FUNC(8, 512, 8, 16, 16, 1, false)
HGEMM_DESC_FUNC(16, 256, 8, 16, 16, 1, false)
HGEMM_DESC_FUNC(8, 128, 8, 16, 16, 4, false)
HGEMM_DESC_FUNC(32, 256, 8, 32, 16, 1, false)
HGEMM_DESC_FUNC(16, 128, 8, 16, 16, 1, false)
HGEMM_DESC_FUNC(8, 256, 8, 32, 16, 4, false)
HGEMM_DESC_FUNC(8, 512, 8, 32, 16, 2, false)
HGEMM_DESC_FUNC(256, 256, 32, 64, 32, 1, false)
HGEMM_DESC_FUNC(8, 128, 8, 16, 32, 4, false)
HGEMM_DESC_FUNC(32, 128, 8, 32, 32, 2, false)
HGEMM_DESC_FUNC(32, 64, 8, 16, 32, 2, false)
HGEMM_DESC_FUNC(256, 256, 32, 64, 16, 1, false)
HGEMM_DESC_FUNC(64, 128, 64, 16, 32, 4, false)

#define HGEMM_NUM_POLICY 14
#define HGEMM_ENUMERATE_FUNC(                                \
    HEAD, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HEAD##_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_

int gemm_policies_wg_mnk[HGEMM_NUM_POLICY][3]{
    {32, 64, 16}, // 0
    {8, 512, 16}, // 1
    {16, 256, 16}, // 2
    {8, 128, 16}, // 3
    {32, 256, 16}, // 4
    {16, 128, 16}, // 5
    {8, 256, 16}, // 6
    {8, 512, 16}, // 7
    {256, 256, 32}, // 8
    {8, 128, 32}, // 9
    {32, 128, 32}, // 10
    {32, 64, 32}, // 11
    {256, 256, 16}, // 12
    {64, 128, 32}, // 13
};

#define HGEMM_ENUMERATE_ALL_FUNC(HEAD)                            \
  HGEMM_ENUMERATE_FUNC(HEAD, 32, 64, 8, 16, 16, 2, true),         \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 512, 8, 16, 16, 1, true),     \
      HGEMM_ENUMERATE_FUNC(HEAD, 16, 256, 8, 16, 16, 1, true),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 128, 8, 16, 16, 4, true),     \
      HGEMM_ENUMERATE_FUNC(HEAD, 32, 256, 8, 32, 16, 1, true),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 16, 128, 8, 16, 16, 1, true),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 256, 8, 32, 16, 4, true),     \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 512, 8, 32, 16, 2, true),     \
      HGEMM_ENUMERATE_FUNC(HEAD, 256, 256, 32, 64, 32, 1, true),  \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 128, 8, 16, 32, 4, true),     \
      HGEMM_ENUMERATE_FUNC(HEAD, 32, 128, 8, 32, 32, 2, true),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 32, 64, 8, 16, 32, 2, true),     \
      HGEMM_ENUMERATE_FUNC(HEAD, 256, 256, 32, 64, 16, 1, true),  \
      HGEMM_ENUMERATE_FUNC(HEAD, 64, 128, 64, 16, 32, 4, true),   \
      HGEMM_ENUMERATE_FUNC(HEAD, 32, 64, 8, 16, 16, 2, false),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 512, 8, 16, 16, 1, false),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 16, 256, 8, 16, 16, 1, false),   \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 128, 8, 16, 16, 4, false),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 32, 256, 8, 32, 16, 1, false),   \
      HGEMM_ENUMERATE_FUNC(HEAD, 16, 128, 8, 16, 16, 1, false),   \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 256, 8, 32, 16, 4, false),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 512, 8, 32, 16, 2, false),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 256, 256, 32, 64, 32, 1, false), \
      HGEMM_ENUMERATE_FUNC(HEAD, 8, 128, 8, 16, 32, 4, false),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 32, 128, 8, 32, 32, 2, false),   \
      HGEMM_ENUMERATE_FUNC(HEAD, 32, 64, 8, 16, 32, 2, false),    \
      HGEMM_ENUMERATE_FUNC(HEAD, 256, 256, 32, 64, 16, 1, false), \
      HGEMM_ENUMERATE_FUNC(HEAD, 64, 128, 64, 16, 32, 4, false),

void (*hgemm_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm)};

void (*hgemm_bias_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_bias)};

void (*hgemm_bias_res_res_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_bias_res_res)};

void (*hgemm_bias_gelu_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_bias_gelu)};

void (*hgemm_resmul_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_resmul)};

void (*hgemm_silu_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_silu)};

void (*hgemm_res_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_res)};

void (*hgemm_qkv_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_qkv)};

void (*hgemm_qkv_bias_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_qkv_bias)};

void (*hgemm_bias_res_policies[2 * HGEMM_NUM_POLICY])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_ALL_FUNC(hgemm_bias_res)};

struct gemm_cfg_meta {
  float wg_eff;
  float num_ss;
  float aspect_r;
  int idx;
  int ks;
};

inline int select_gemm_config(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major,
    const int TOTAL_SS = 64) {
  std::vector<gemm_cfg_meta> metas;
  for (int i = 0; i < HGEMM_NUM_POLICY; i++) {
    gemm_cfg_meta meta;
    int wg_m = gemm_policies_wg_mnk[i][0];
    int wg_n = gemm_policies_wg_mnk[i][1];
    int ms = (m + wg_m - 1) / wg_m;
    int ns = (n + wg_n - 1) / wg_n;
    meta.num_ss = ms * ns;
    int vm = m > wg_m ? wg_m : m;
    int vn = n > wg_n ? wg_n : n;
    meta.wg_eff = (float)vm * vn / (float)wg_m / (float)wg_n;
    meta.idx = i;
    meta.aspect_r = std::max((float)wg_m / wg_n, (float)wg_n / wg_m);
    meta.ks = gemm_policies_wg_mnk[i][2];
    metas.push_back(meta);
  }
  std::sort(metas.begin(), metas.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.num_ss != rhs.num_ss)
      return lhs.num_ss < rhs.num_ss;
    else if (lhs.wg_eff != rhs.wg_eff)
      return lhs.wg_eff > rhs.wg_eff;
    else if (lhs.aspect_r != rhs.aspect_r)
      return lhs.aspect_r < rhs.aspect_r;
    else
      return lhs.ks < rhs.ks;
  });
  int idx;
  for (int i = 0; i < metas.size(); i++) {
    idx = metas[i].idx;
    if (metas[i].num_ss >= TOTAL_SS)
      break;
  }
  return is_b_row_major ? idx : idx + HGEMM_NUM_POLICY;
}

} // namespace xetla
} // namespace xpu
