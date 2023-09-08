#pragma once

#include <sycl/sycl.hpp>

namespace xpu {
namespace xetla {

#define HGEMM_FUNC_NAME(                                     \
    HEAD, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HEAD##_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_

#define HGEMM_ADDMM_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  HGEMM_FUNC_NAME(                                                          \
      hgemm_addmm, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
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
  void HGEMM_ADDMM_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)(    \
      sycl::queue & queue,                                                     \
      sycl::half * out,                                                        \
      const sycl::half* res,                                                   \
      const sycl::half* a,                                                     \
      const sycl::half* b,                                                     \
      const int m,                                                             \
      const int n,                                                             \
      const int k,                                                             \
      const float alpha,                                                       \
      const float beta);                                                       \
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
#define HGEMM_NUM_POLICIES 26
#define HGEMM_ENUMERATE_POLICIES_(_, B_ROW_MAJOR, T) \
  _(8, 64, 8, 16, 32, 8, B_ROW_MAJOR)T      \
  _(8, 128, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(8, 128, 8, 16, 32, 4, B_ROW_MAJOR)T     \
  _(8, 256, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(8, 512, 8, 16, 16, 1, B_ROW_MAJOR)T     \
  _(16, 64, 16, 16, 16, 8, B_ROW_MAJOR)T    \
  _(16, 256, 8, 16, 16, 1, B_ROW_MAJOR)T    \
  _(16, 256, 16, 16, 16, 2, B_ROW_MAJOR)T   \
  _(16, 512, 16, 16, 16, 1, B_ROW_MAJOR)T   \
  _(32, 64, 32, 16, 16, 8, B_ROW_MAJOR)T    \
  _(32, 64, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(32, 128, 32, 16, 16, 4, B_ROW_MAJOR)T   \
  _(32, 256, 32, 16, 16, 2, B_ROW_MAJOR)T   \
  _(32, 512, 32, 16, 16, 1, B_ROW_MAJOR)T   \
  _(64, 128, 64, 16, 16, 4, B_ROW_MAJOR)T   \
  _(64, 256, 64, 16, 16, 2, B_ROW_MAJOR)T   \
  _(64, 512, 64, 16, 16, 1, B_ROW_MAJOR)T   \
  _(128, 128, 32, 32, 32, 2, B_ROW_MAJOR)T  \
  _(128, 256, 64, 16, 16, 1, B_ROW_MAJOR)T  \
  _(128, 512, 64, 32, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 64, 32, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 32, 64, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 32, 64, 32, 1, B_ROW_MAJOR)T  \
  _(128, 64, 16, 16, 64, 1, B_ROW_MAJOR)T   \
  _(128, 128, 16, 32, 64, 1, B_ROW_MAJOR)T  \
  _(128, 256, 32, 32, 16, 1, B_ROW_MAJOR)T
// clang-format on

#define HGEMM_ENUMERATE_POLICIES(_)    \
  HGEMM_ENUMERATE_POLICIES_(_, true, ) \
  HGEMM_ENUMERATE_POLICIES_(_, false, )

#define HGEMM_ENUMERATE_POLICIES_COMMA(_)         \
  HGEMM_ENUMERATE_POLICIES_(_, true, HGEMM_COMMA) \
  HGEMM_ENUMERATE_POLICIES_(_, false, HGEMM_COMMA)

HGEMM_ENUMERATE_POLICIES(HGEMM_ENUMERATE_FUNC_DESCS)

#define HGEMM_POLICY_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  "_" #WG_M "x" #WG_N "_" #SG_M "x" #SG_N "x" #SG_K "_" #SLM_KS              \
  "_" #B_ROW_MAJOR "_"
#define HGEMM_POLICY_NAME_SYMBOL(                      \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  _##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_
const char* hgemm_policy_names[2 * HGEMM_NUM_POLICIES] = {
    HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_POLICY_NAME)};
enum hgemm_policy {
  NONE = -1,
  HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_POLICY_NAME_SYMBOL)
};
int hgemm_get_policy(hgemm_policy name, bool is_b_row_major) {
  int idx = static_cast<int>(name);
  return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
}

#define HGEMM_ENUMERATE_FUNC_TRAITS(                   \
    WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  { WG_M, WG_N }
int hgemm_policies_wg_mnk[2 * HGEMM_NUM_POLICIES][2]{
    HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_ENUMERATE_FUNC_TRAITS)};

void (*hgemm_addmm_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_ADDMM_FUNC)};

void (*hgemm_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_FUNC)};

void (*hgemm_bias_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_FUNC)};

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
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RES_RES_FUNC)};

void (*hgemm_bias_gelu_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_GELU_FUNC)};

void (*hgemm_resmul_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RESMUL_FUNC)};

void (*hgemm_silu_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_SILU_FUNC)};

void (*hgemm_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RES_FUNC)};

void (*hgemm_qkv_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&,
    sycl::half*,
    sycl::half*,
    sycl::half*,
    const sycl::half*,
    const sycl::half*,
    const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_FUNC)};

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
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_BIAS_FUNC)};

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
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RES_FUNC)};

std::unordered_map<std::string, int> special_mnk2policy = {
    {"m=1, n=4096, k=16384", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=1, n=16384, k=4096", hgemm_policy::_8x512_8x16x16_1_true_},
    {"m=1, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=4, n=4096, k=16384", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=4, n=16384, k=4096", hgemm_policy::_8x512_8x16x16_1_true_},
    {"m=4, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=32, n=4096, k=16384", hgemm_policy::_32x64_8x16x16_2_true_},
    {"m=32, n=16384, k=4096", hgemm_policy::_32x512_32x16x16_1_true_},
    {"m=32, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=33, n=4096, k=16384", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=33, n=16384, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=33, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=64, n=4096, k=16384", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=64, n=16384, k=4096", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=64, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=65, n=4096, k=16384", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=65, n=16384, k=4096", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=65, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=128, n=4096, k=16384", hgemm_policy::_128x128_32x32x32_2_true_},
    {"m=128, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=128, n=4096, k=4096", hgemm_policy::_128x128_32x32x32_2_true_},
    {"m=130, n=4096, k=16384", hgemm_policy::_128x128_32x32x32_2_true_},
    {"m=130, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=130, n=4096, k=4096", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=256, n=4096, k=16384", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=256, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=256, n=4096, k=4096", hgemm_policy::_128x128_32x32x32_2_true_},
    {"m=512, n=4096, k=16384", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=512, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=512, n=4096, k=4096", hgemm_policy::_128x128_32x32x32_2_true_},
    {"m=513, n=4096, k=16384", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=513, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=513, n=4096, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=1024, n=4096, k=16384", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1024, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1024, n=4096, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1028, n=4096, k=16384", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1028, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1028, n=4096, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=2016, n=4096, k=16384", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=2016, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=2016, n=4096, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1, n=50400, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=1, n=50272, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=4, n=50400, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=4, n=50272, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=1, n=250880, k=4096", hgemm_policy::_32x64_8x16x16_2_true_},
    {"m=4, n=250880, k=4096", hgemm_policy::_32x64_8x16x16_2_true_},
    {"m=1, n=11008, k=4096", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=4, n=11008, k=4096", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=32, n=11008, k=4096", hgemm_policy::_64x256_64x16x16_2_true_},
    {"m=64, n=11008, k=4096", hgemm_policy::_64x256_64x16x16_2_true_},
    {"m=128, n=11008, k=4096", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=256, n=11008, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=512, n=11008, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1024, n=11008, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=2016, n=11008, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1, n=32000, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=4, n=32000, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1, n=13824, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1, n=5120, k=5120", hgemm_policy::_8x128_8x16x16_2_true_},
    {"m=4, n=13824, k=5120", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=4, n=5120, k=5120", hgemm_policy::_8x128_8x16x16_2_true_},
    {"m=32, n=13824, k=5120", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=32, n=5120, k=5120", hgemm_policy::_32x128_32x16x16_4_true_},
    {"m=64, n=13824, k=5120", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=64, n=5120, k=5120", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=128, n=13824, k=5120", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=128, n=5120, k=5120", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=256, n=13824, k=5120", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=256, n=5120, k=5120", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=512, n=13824, k=5120", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=512, n=5120, k=5120", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1024, n=13824, k=5120", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1024, n=5120, k=5120", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=2016, n=13824, k=5120", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=2016, n=5120, k=5120", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1, n=32000, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=4, n=32000, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1, n=7168, k=14336", hgemm_policy::_8x256_8x16x16_2_true_},
    {"m=1, n=1792, k=14336", hgemm_policy::_16x64_16x16x16_8_true_},
    {"m=4, n=7168, k=14336", hgemm_policy::_8x256_8x16x16_2_true_},
    {"m=4, n=1792, k=14336", hgemm_policy::_16x64_16x16x16_8_true_},
    {"m=32, n=7168, k=14336", hgemm_policy::_32x256_32x16x16_2_true_},
    {"m=32, n=1792, k=14336", hgemm_policy::_16x64_16x16x16_8_true_},
    {"m=1, n=14336, k=7168", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=1, n=14336, k=1792", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=4, n=14336, k=7168", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=4, n=14336, k=1792", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=32, n=14336, k=7168", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=32, n=14336, k=1792", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=1, n=250880, k=1792", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=1, n=2048, k=8192", hgemm_policy::_8x64_8x16x32_8_true_},
    {"m=1, n=3584, k=7168", hgemm_policy::_32x64_8x16x16_2_true_},
    {"m=1, n=7168, k=3584", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=1, n=7168, k=8192", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=1, n=8192, k=2048", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=1, n=8192, k=7168", hgemm_policy::_8x256_8x16x16_2_true_},
    {"m=1, n=256, k=8192", hgemm_policy::_16x64_16x16x16_8_true_},
    {"m=1, n=32000, k=2048", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=4, n=2048, k=8192", hgemm_policy::_8x64_8x16x32_8_true_},
    {"m=4, n=3584, k=7168", hgemm_policy::_32x64_8x16x16_2_true_},
    {"m=4, n=7168, k=3584", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=4, n=7168, k=8192", hgemm_policy::_8x256_8x16x16_2_true_},
    {"m=4, n=8192, k=2048", hgemm_policy::_8x256_8x16x16_2_true_},
    {"m=4, n=8192, k=7168", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=4, n=256, k=8192", hgemm_policy::_16x64_16x16x16_8_true_},
    {"m=4, n=32000, k=2048", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1024, n=2048, k=8192", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=1024, n=7168, k=8192", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1024, n=8192, k=2048", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=1024, n=8192, k=7168", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=256, k=8192", hgemm_policy::_32x128_32x16x16_4_true_},
};

std::unordered_map<std::string, int> special_qkv_mnk2policy = {
    {"m=1, n=4096, k=16384", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=1, n=16384, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=4, n=4096, k=16384", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=4, n=16384, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=4, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=32, n=4096, k=16384", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=32, n=16384, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=32, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=33, n=4096, k=16384", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=33, n=16384, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=33, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=64, n=4096, k=16384", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=64, n=16384, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=64, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=65, n=4096, k=16384", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=65, n=16384, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=65, n=4096, k=4096", hgemm_policy::_128x64_16x16x64_1_true_},
    {"m=128, n=4096, k=16384", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=128, n=16384, k=4096", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=128, n=4096, k=4096", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=130, n=4096, k=16384", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=130, n=16384, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=130, n=4096, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=256, n=4096, k=16384", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=256, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=256, n=4096, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=512, n=4096, k=16384", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=512, n=16384, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=512, n=4096, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=513, n=4096, k=16384", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=513, n=16384, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=513, n=4096, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=1024, n=4096, k=16384", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=16384, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=4096, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1028, n=4096, k=16384", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1028, n=16384, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=1028, n=4096, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=2016, n=4096, k=16384", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=2016, n=16384, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=2016, n=4096, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1, n=50400, k=4096", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=1, n=50272, k=4096", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=4, n=50400, k=4096", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=4, n=50272, k=4096", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=1, n=11008, k=4096", hgemm_policy::_32x64_8x16x16_2_true_},
    {"m=4, n=11008, k=4096", hgemm_policy::_32x64_8x16x16_2_true_},
    {"m=32, n=11008, k=4096", hgemm_policy::_32x64_8x16x16_2_true_},
    {"m=64, n=11008, k=4096", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=128, n=11008, k=4096", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=256, n=11008, k=4096", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=512, n=11008, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=11008, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=2016, n=11008, k=4096", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1, n=32000, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=4, n=32000, k=4096", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1, n=13824, k=5120", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=1, n=5120, k=5120", hgemm_policy::_128x512_64x32x16_1_true_},
    {"m=4, n=13824, k=5120", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=4, n=5120, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=32, n=13824, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=32, n=5120, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=64, n=13824, k=5120", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=64, n=5120, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=128, n=13824, k=5120", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=128, n=5120, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=256, n=13824, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=256, n=5120, k=5120", hgemm_policy::_256x256_64x32x16_1_true_},
    {"m=512, n=13824, k=5120", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=512, n=5120, k=5120", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=13824, k=5120", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=5120, k=5120", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=2016, n=13824, k=5120", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=2016, n=5120, k=5120", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1, n=32000, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=4, n=32000, k=5120", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1, n=7168, k=14336", hgemm_policy::_32x512_32x16x16_1_true_},
    {"m=1, n=1792, k=14336", hgemm_policy::_8x128_8x16x16_2_true_},
    {"m=4, n=7168, k=14336", hgemm_policy::_64x512_64x16x16_1_true_},
    {"m=4, n=1792, k=14336", hgemm_policy::_8x128_8x16x16_2_true_},
    {"m=32, n=7168, k=14336", hgemm_policy::_64x512_64x16x16_1_true_},
    {"m=32, n=1792, k=14336", hgemm_policy::_64x128_64x16x16_4_true_},
    {"m=1, n=14336, k=7168", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=1, n=14336, k=1792", hgemm_policy::_128x256_64x16x16_1_true_},
    {"m=4, n=14336, k=7168", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=4, n=14336, k=1792", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=32, n=14336, k=7168", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=32, n=14336, k=1792", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=1, n=250880, k=1792", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1, n=2048, k=8192", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=1, n=3584, k=7168", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=1, n=7168, k=3584", hgemm_policy::_64x512_64x16x16_1_true_},
    {"m=1, n=7168, k=8192", hgemm_policy::_64x512_64x16x16_1_true_},
    {"m=1, n=8192, k=2048", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=1, n=8192, k=7168", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=1, n=256, k=8192", hgemm_policy::_16x64_16x16x16_8_true_},
    {"m=1, n=32000, k=2048", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=4, n=2048, k=8192", hgemm_policy::_128x128_16x32x64_1_true_},
    {"m=4, n=3584, k=7168", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=4, n=7168, k=3584", hgemm_policy::_64x512_64x16x16_1_true_},
    {"m=4, n=7168, k=8192", hgemm_policy::_64x512_64x16x16_1_true_},
    {"m=4, n=8192, k=2048", hgemm_policy::_16x256_8x16x16_1_true_},
    {"m=4, n=8192, k=7168", hgemm_policy::_128x256_32x32x16_1_true_},
    {"m=4, n=256, k=8192", hgemm_policy::_16x64_16x16x16_8_true_},
    {"m=4, n=32000, k=2048", hgemm_policy::_256x256_32x64x16_1_true_},
    {"m=1024, n=2048, k=8192", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=7168, k=8192", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=8192, k=2048", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=8192, k=7168", hgemm_policy::_256x256_32x64x32_1_true_},
    {"m=1024, n=256, k=8192", hgemm_policy::_128x128_32x32x32_2_true_},
};

inline int select_gemm_qkv_special_config(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major) {
  std::ostringstream traits;
  traits << "m=" << m << ", n=" << n << ", k=" << k;
  std::string traits_str = traits.str();
  auto it = special_qkv_mnk2policy.find(traits_str);
  if (it != special_qkv_mnk2policy.end()) {
    int idx = it->second;
    return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
  }
  return -1;
}

inline int select_gemm_special_config(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major) {
  std::ostringstream traits;
  traits << "m=" << m << ", n=" << n << ", k=" << k;
  std::string traits_str = traits.str();
  auto it = special_mnk2policy.find(traits_str);
  if (it != special_mnk2policy.end()) {
    int idx = it->second;
    return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
  }

  if (n == 4096 && m <= 128) {
    return hgemm_get_policy(
        hgemm_policy::_128x64_16x16x64_1_true_, is_b_row_major);
  } else if (m >= 64) {
    if (m <= 512 && n <= 5120) {
      return hgemm_get_policy(
          hgemm_policy::_128x128_32x32x32_2_true_, is_b_row_major);
    } else {
      return hgemm_get_policy(
          hgemm_policy::_256x256_64x32x16_1_true_, is_b_row_major);
    }
  }

  if (m <= 8 && n <= 4096) {
    return hgemm_get_policy(
        hgemm_policy::_32x64_8x16x16_2_true_, is_b_row_major);
  } else if (m <= 16) {
    return -1; // let auto-config choose
  } else if (m <= 256 && n >= 16384) {
    return hgemm_get_policy(
        hgemm_policy::_256x256_32x64x16_1_true_, is_b_row_major);
  } else if (((m + 255) / 256) * ((n + 255) / 256) >= 64) {
    return hgemm_get_policy(
        hgemm_policy::_256x256_32x64x32_1_true_, is_b_row_major);
  } else if (m <= 128 && n == 4096) {
    return hgemm_get_policy(
        hgemm_policy::_128x64_16x16x64_1_true_, is_b_row_major);
  } else if (m <= 256 && n == 4096) {
    return hgemm_get_policy(
        hgemm_policy::_128x128_16x32x64_1_true_, is_b_row_major);
  } else if (m <= 512 && n == 4096) {
    return hgemm_get_policy(
        hgemm_policy::_128x256_32x32x16_1_true_, is_b_row_major);
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
  struct gemm_cfg_meta {
    float wg_eff;
    float num_ss;
    float aspect_r;
    int idx;
  };
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
