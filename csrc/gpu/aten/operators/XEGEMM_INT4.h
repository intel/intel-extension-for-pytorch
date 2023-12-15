#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"
#include "xetla/GEMM_INT4.h"

using namespace xpu::xetla;

#define RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS) \
  char str__[100];                                                        \
  sprintf(                                                                \
      str__,                                                              \
      "%s(%d, %d, %d, %d, %d, %d, %d, %d, %d, %d)",                       \
      "" #F,                                                              \
      WG_M,                                                               \
      WG_N,                                                               \
      SG_M,                                                               \
      SG_N,                                                               \
      SG_K,                                                               \
      GZ,                                                                 \
      SLM_KS,                                                             \
      m_,                                                                 \
      n_,                                                                 \
      k_);                                                                \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

#define HGEMM_INT4_DISPATCH(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)  \
  {                                                                       \
    RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)     \
    F<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS>(              \
        q,                                                                \
        reinterpret_cast<sycl::half*>(outputs_[0]->data_ptr<scalar_t>()), \
        reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),      \
        weight_->data_ptr<uint8_t>(),                                     \
        weight_zp_->data_ptr<uint8_t>(),                                  \
        reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()), \
        m_,                                                               \
        n_,                                                               \
        k_);                                                              \
  }

#define HGEMM_INT4_BIAS_DISPATCH(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS) \
  {                                                                           \
    RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)         \
    F<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS>(                  \
        q,                                                                    \
        reinterpret_cast<sycl::half*>(outputs_[0]->data_ptr<scalar_t>()),     \
        reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),          \
        weight_->data_ptr<uint8_t>(),                                         \
        weight_zp_->data_ptr<uint8_t>(),                                      \
        reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),     \
        reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()),   \
        m_,                                                                   \
        n_,                                                                   \
        k_);                                                                  \
  }

#define HGEMM_INT4_BIAS_RES_RES_DISPATCH(                                   \
    F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)                            \
  {                                                                         \
    RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)       \
    F<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS>(                \
        q,                                                                  \
        reinterpret_cast<sycl::half*>(outputs_[0]->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),        \
        weight_->data_ptr<uint8_t>(),                                       \
        weight_zp_->data_ptr<uint8_t>(),                                    \
        reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
        reinterpret_cast<sycl::half*>(epilogues_[1]->data_ptr<scalar_t>()), \
        reinterpret_cast<sycl::half*>(epilogues_[2]->data_ptr<scalar_t>()), \
        m_,                                                                 \
        n_,                                                                 \
        k_);                                                                \
  }

#define HGEMM_INT4_BIAS_GELU_DISPATCH(                                      \
    F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)                            \
  {                                                                         \
    RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)       \
    F<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS>(                \
        q,                                                                  \
        reinterpret_cast<sycl::half*>(outputs_[0]->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),        \
        weight_->data_ptr<uint8_t>(),                                       \
        weight_zp_->data_ptr<uint8_t>(),                                    \
        reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
        m_,                                                                 \
        n_,                                                                 \
        k_);                                                                \
  }

#define HGEMM_INT4_RES_DISPATCH(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS) \
  {                                                                          \
    RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)        \
    F<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS>(                 \
        q,                                                                   \
        reinterpret_cast<sycl::half*>(outputs_[0]->data_ptr<scalar_t>()),    \
        reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),         \
        weight_->data_ptr<uint8_t>(),                                        \
        weight_zp_->data_ptr<uint8_t>(),                                     \
        reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),    \
        reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()),  \
        m_,                                                                  \
        n_,                                                                  \
        k_);                                                                 \
  }

#define HGEMM_INT4_RESMUL_DISPATCH(                                         \
    F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)                            \
  {                                                                         \
    RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)       \
    F<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS>(                \
        q,                                                                  \
        reinterpret_cast<sycl::half*>(outputs_[0]->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),        \
        weight_->data_ptr<uint8_t>(),                                       \
        weight_zp_->data_ptr<uint8_t>(),                                    \
        reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
        m_,                                                                 \
        n_,                                                                 \
        k_);                                                                \
  }

#define HGEMM_INT4_QKV_DISPATCH(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS) \
  {                                                                          \
    RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)        \
    F<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS>(                 \
        q,                                                                   \
        reinterpret_cast<sycl::half*>(outputs_[0]->data_ptr<scalar_t>()),    \
        reinterpret_cast<sycl::half*>(outputs_[1]->data_ptr<scalar_t>()),    \
        reinterpret_cast<sycl::half*>(outputs_[2]->data_ptr<scalar_t>()),    \
        reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),         \
        weight_->data_ptr<uint8_t>(),                                        \
        weight_zp_->data_ptr<uint8_t>(),                                     \
        reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),    \
        m_,                                                                  \
        n_,                                                                  \
        k_);                                                                 \
  }

#define HGEMM_INT4_QKV_BIAS_DISPATCH(                                       \
    F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)                            \
  {                                                                         \
    RECORD_FUNCTION_IMPL(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)       \
    F<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS>(                \
        q,                                                                  \
        reinterpret_cast<sycl::half*>(outputs_[0]->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(outputs_[1]->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(outputs_[2]->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),        \
        weight_->data_ptr<uint8_t>(),                                       \
        weight_zp_->data_ptr<uint8_t>(),                                    \
        reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),   \
        reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
        m_,                                                                 \
        n_,                                                                 \
        k_);                                                                \
  }

#define HGEMM_INT4_COMMON_DISPATCH_IMPL(                                    \
    DISPATCHER, F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)                \
  DISPATCHER(F, WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS)

#define HGEMM_INT4_COMMON_DISPATCH(                                          \
    WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS, ARCH)                          \
  {                                                                          \
    if (num_epilogues_ == 0 && ARCH == 1)                                    \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_DISPATCH,                                               \
          hgemm_wint4_pvc,                                                   \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (num_epilogues_ == 0 && ARCH == 0)                               \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_DISPATCH,                                               \
          hgemm_wint4_arc,                                                   \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == BIAS && ARCH == 1)  \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_BIAS_DISPATCH,                                          \
          hgemm_bias_wint4_pvc,                                              \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == BIAS && ARCH == 0)  \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_BIAS_DISPATCH,                                          \
          hgemm_bias_wint4_arc,                                              \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (                                                                \
        num_epilogues_ == 3 && epilogue_type_[0] == BIAS &&                  \
        epilogue_type_[1] == RES_ADD && epilogue_type_[2] == RES_ADD &&      \
        ARCH == 1)                                                           \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_BIAS_RES_RES_DISPATCH,                                  \
          hgemm_bias_res_res_wint4_pvc,                                      \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (                                                                \
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&                  \
        epilogue_type_[1] == GELU && ARCH == 1)                              \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_BIAS_GELU_DISPATCH,                                     \
          hgemm_bias_gelu_wint4_pvc,                                         \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_ADD             \
          && ARCH == 1)                                                      \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_RES_DISPATCH,                                           \
          hgemm_res_wint4_pvc,                                               \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_MUL             \
          && ARCH == 1)                                                      \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_RESMUL_DISPATCH,                                        \
          hgemm_mul_wint4_pvc,                                               \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == SPLIT3 &&           \
          ARCH == 1)                                                         \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_QKV_DISPATCH,                                           \
          hgemm_qkv_wint4_pvc,                                               \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
    else if (                                                                \
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&                  \
        epilogue_type_[1] == SPLIT3 && ARCH == 1)                            \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                       \
          HGEMM_INT4_QKV_BIAS_DISPATCH,                                      \
          hgemm_qkv_bias_wint4_pvc,                                          \
          WG_M,                                                              \
          WG_N,                                                              \
          SG_M,                                                              \
          SG_N,                                                              \
          SG_K,                                                              \
          GZ,                                                                \
          SLM_KS)                                                            \
  }

template <
    int wg_m_,
    int wg_n_,
    int sg_m_,
    int sg_n_,
    int sg_k_,
    int gz_,
    int slm_ks_,
    int max_m_,
    int max_n_,
    int max_k_,
    int arch_>
struct GemmWint4Config {
  static constexpr int wg_m = wg_m_;
  static constexpr int wg_n = wg_n_;
  static constexpr int sg_m = sg_m_;
  static constexpr int sg_n = sg_n_;
  static constexpr int sg_k = sg_k_;
  static constexpr int gz = gz_;
  static constexpr int slm_ks = slm_ks_;
  static constexpr int max_m = max_m_;
  static constexpr int max_n = max_n_;
  static constexpr int max_k = max_k_;
  static constexpr int arch = arch_;

  static bool less_than(int m, int n, int k, int group_size) {
    if (gz < group_size)
      return true;
    if (gz == group_size && max_n < n)
      return true;
    if (gz == group_size && max_n == n && max_k < k)
      return true;
    if (gz == group_size && max_n == n && max_k == k && max_m < m)
      return true;
    return false;
  }
};

#define MAX_INT std::numeric_limits<int>::max()

// clang-format off
#define ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_PVC(gz)                       \
  GemmWint4Config<8, 64, 8, 16, 64, gz, 8, 8, 4096, 4096, 1>,               \
  GemmWint4Config<16, 64, 16, 16, 32, gz, 8, 16, 4096, 4096, 1>,            \
  GemmWint4Config<32, 64, 32, 16, 32, gz, 8, 32, 4096, 4096, 1>,            \
  GemmWint4Config<32, 128, 32, 16, 32, gz, 4, 64, 4096, 4096, 1>,           \
  GemmWint4Config<64, 128, 64, 16, 32, gz, 4, 384, 4096, 4096, 1>,          \
  GemmWint4Config<128, 256, 64, 16, 32, gz, 1, MAX_INT, 4096, 4096, 1>,     \
  GemmWint4Config<8, 64, 8, 16, 64, gz, 8, 8, 4096, 16384, 1>,              \
  GemmWint4Config<16, 64, 16, 16, 32, gz, 8, 16, 4096, 16384, 1>,           \
  GemmWint4Config<32, 64, 32, 16, 32, gz, 8, 32, 4096, 16384, 1>,           \
  GemmWint4Config<32, 128, 32, 16, 32, gz, 4, 64, 4096, 16384, 1>,          \
  GemmWint4Config<64, 128, 64, 16, 32, gz, 4, 384, 4096, 16384, 1>,         \
  GemmWint4Config<128, 256, 64, 16, 32, gz, 1, MAX_INT, 4096, 16384, 1>,    \
  GemmWint4Config<8, 64, 8, 16, 64, gz, 8, 8, 4096, MAX_INT, 1>,            \
  GemmWint4Config<16, 64, 16, 16, 32, gz, 8, 16, 4096, MAX_INT, 1>,         \
  GemmWint4Config<32, 64, 32, 16, 32, gz, 8, 32, 4096, MAX_INT, 1>,         \
  GemmWint4Config<32, 128, 32, 16, 32, gz, 4, 64, 4096, MAX_INT, 1>,        \
  GemmWint4Config<64, 128, 64, 16, 32, gz, 4, 384, 4096, MAX_INT, 1>,       \
  GemmWint4Config<128, 256, 64, 16, 32, gz, 1, MAX_INT, 4096, MAX_INT, 1>,  \
  GemmWint4Config<8, 256, 8, 16, 32, gz, 2, 8, 16384, 4096, 1>,             \
  GemmWint4Config<16, 256, 16, 16, 32, gz, 2, 16, 16384, 4096, 1>,          \
  GemmWint4Config<32, 256, 32, 16, 32, gz, 2, 32, 16384, 4096, 1>,          \
  GemmWint4Config<64, 256, 64, 16, 32, gz, 2, 64, 16384, 4096, 1>,          \
  GemmWint4Config<64, 128, 64, 16, 32, gz, 4, 384, 16384, 4096, 1>,         \
  GemmWint4Config<128, 256, 64, 16, 32, gz, 1, MAX_INT, 16384, 4096, 1>,    \
  GemmWint4Config<8, 256, 8, 16, 32, gz, 2, 8, 16384, MAX_INT, 1>,          \
  GemmWint4Config<16, 256, 16, 16, 32, gz, 2, 16, 16384, MAX_INT, 1>,       \
  GemmWint4Config<32, 256, 32, 16, 32, gz, 2, 32, 16384, MAX_INT, 1>,       \
  GemmWint4Config<64, 256, 64, 16, 32, gz, 2, 64, 16384, MAX_INT, 1>,       \
  GemmWint4Config<64, 128, 64, 16, 32, gz, 4, 384, 16384, MAX_INT, 1>,      \
  GemmWint4Config<128, 256, 64, 16, 32, gz, 1, MAX_INT, 16384, MAX_INT, 1>, \
  GemmWint4Config<8, 512, 8, 16, 32, gz, 1, 8, 50416, 4096, 1>,             \
  GemmWint4Config<16, 512, 16, 16, 32, gz, 1, 16, 50416, 4096, 1>,          \
  GemmWint4Config<32, 512, 32, 16, 32, gz, 1, 32, 50416, 4096, 1>,          \
  GemmWint4Config<64, 512, 64, 16, 32, gz, 1, 64, 50416, 4096, 1>,          \
  GemmWint4Config<64, 128, 64, 16, 32, gz, 4, 384, 50416, 4096, 1>,         \
  GemmWint4Config<128, 512, 64, 32, 32, gz, 1, MAX_INT, 50416, 4096, 1>,    \
  GemmWint4Config<8, 512, 8, 16, 32, gz, 1, 8, MAX_INT, MAX_INT, 1>,        \
  GemmWint4Config<16, 512, 16, 16, 32, gz, 1, 16, MAX_INT, MAX_INT, 1>,     \
  GemmWint4Config<32, 512, 32, 16, 32, gz, 1, 32, MAX_INT, MAX_INT, 1>,     \
  GemmWint4Config<64, 512, 64, 16, 32, gz, 1, 64, MAX_INT, MAX_INT, 1>,     \
  GemmWint4Config<64, 128, 64, 16, 32, gz, 4, 384, MAX_INT, MAX_INT, 1>,    \
  GemmWint4Config<128, 512, 64, 32, 32, gz, 1, MAX_INT, MAX_INT, MAX_INT, 1>

#define ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_ARC(gz)                       \
  GemmWint4Config<8, 64, 8, 16, 16, gz, 8, MAX_INT, MAX_INT, MAX_INT, 0>
// clang-format on

#define ORDERED_GEMM_WINT4_CONFIG_SET_ARC       \
  ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_ARC(16), \
  ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_ARC(64), \
  ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_ARC(128), \
  ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_ARC(256), \
  ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_ARC(512), \
  ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_ARC(1024)

#define ORDERED_GEMM_WINT4_CONFIG_SET_PVC       \
  ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_PVC(0), \
      ORDERED_GEMM_WINT4_CONFIG_SET_WITH_GZ_PVC(128)

inline Tensor resize_as_mat1(const Tensor& mat1, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = mat1.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

class HGEMMXetla_INT4 final {
 public:
  enum EpilogueType {
    BIAS = 0,
    RES_ADD,
    GELU,
    RES_MUL,
    SILU,
    SCALED_RES_ADD,
    SPLIT3,
  };

 private:
  enum {
    MAX_EPILOGUES = 4,
  };
  Tensor *input_, *weight_, *weight_scl_, *weight_zp_;
  std::vector<Tensor*> outputs_;
  Tensor* epilogues_[MAX_EPILOGUES];
  EpilogueType epilogue_type_[MAX_EPILOGUES];
  float pf32[MAX_EPILOGUES];
  int num_epilogues_ = 0;
  bool is_a_row_major_;
  bool is_a_col_major_;
  bool is_b_row_major_;
  bool is_b_col_major_;
  bool fallback_;
  int m_, n_, k_;
  int64_t calib_gz_;
  int64_t arch_ = 1; // 0: ARC, 1: PVC

 public:
  HGEMMXetla_INT4() = default;
  bool fallback() const {
    return fallback_;
  }
  HGEMMXetla_INT4& add_arch(int64_t arch) {
    arch_ = arch;
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_out(const Tensor& output) {
    outputs_.emplace_back(const_cast<Tensor*>(&output));
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_inp(const Tensor& input) {
    input_ = const_cast<Tensor*>(&input);
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_wei(const Tensor& b) {
    weight_ = const_cast<Tensor*>(&b);
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_scl(const Tensor& scales) {
    weight_scl_ = const_cast<Tensor*>(&scales);
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_zp(const Tensor& zero_points) {
    weight_zp_ = const_cast<Tensor*>(&zero_points);
    return *this;
  }
  HGEMMXetla_INT4& add_calib_gz(int64_t calib_gz) {
    calib_gz_ = calib_gz;
    return *this;
  }
  HGEMMXetla_INT4& add_epilogue(const Tensor& t, EpilogueType eptype) {
    epilogues_[num_epilogues_] = const_cast<Tensor*>(&t);
    epilogue_type_[num_epilogues_++] = eptype;
    return *this;
  }
  HGEMMXetla_INT4& add_epilogue(
      const Tensor& t,
      EpilogueType eptype,
      const float x) {
    epilogues_[num_epilogues_] = const_cast<Tensor*>(&t);
    pf32[num_epilogues_] = x;
    epilogue_type_[num_epilogues_++] = eptype;
    return *this;
  }

  HGEMMXetla_INT4& build() {
    fallback_ = true;
    if (input_->scalar_type() != kHalf ||
        (weight_->scalar_type() != kByte &&
         weight_->scalar_type() != kQUInt8) ||
        std::any_of(outputs_.begin(), outputs_.end(), [](Tensor* out) {
          return out->scalar_type() != kHalf;
        }))
      return *this;
    bool has_split3 =
        (epilogue_type_[0] == SPLIT3 ||
         (epilogue_type_[0] == BIAS && epilogue_type_[1] == SPLIT3));
    if (!(input_->dim() == 2 &&
          ((!has_split3 && weight_->dim() == 2) ||
           (has_split3 && weight_->dim() == 3)) &&
          std::all_of(outputs_.begin(), outputs_.end(), [](Tensor* out) {
            return out->dim() == 2;
          })))
      return *this;
    is_a_row_major_ = input_->is_contiguous();
    is_a_col_major_ = input_->transpose(0, 1).is_contiguous();
    is_b_row_major_ = weight_->is_contiguous();
    is_b_col_major_ = weight_->transpose(0, 1).is_contiguous();
    auto a_sizes = input_->sizes();
    auto b_sizes = weight_->sizes();
    m_ = a_sizes[0];
    k_ = a_sizes[1];
    // Normalize calibration group size.
    if ((calib_gz_ == -1 || calib_gz_ == k_) && arch_ == 1)
      calib_gz_ = 0;
    // Check if calibration group size is valid.
    if ((calib_gz_ != 0 && calib_gz_ != 128) && arch_ == 1)
      return *this;
    // Set correct n dim.
    if (has_split3)
      n_ = b_sizes[2] * 2;
    else
      n_ = b_sizes[1] * 2;
    for (int i = 0; i < num_epilogues_; i++) {
      switch (epilogue_type_[i]) {
        case BIAS: {
          bool ck = ((!has_split3 && epilogues_[i]->dim() == 1) ||
                     (has_split3 && epilogues_[i]->dim() == 2)) &&
              epilogues_[i]->is_contiguous();
          ck = ck &&
              ((!has_split3 && epilogues_[i]->sizes()[0] == n_) ||
               (has_split3 && epilogues_[i]->sizes()[1] == n_));
          ck = ck && epilogues_[i]->scalar_type() == kHalf;
          if (!ck)
            return *this;
        } break;
        case RES_MUL:
        case RES_ADD: {
          bool ck = epilogues_[i]->dim() == 2;
          ck = ck && epilogues_[i]->sizes()[0] == m_ &&
              epilogues_[i]->sizes()[1] == n_;
          ck = ck && epilogues_[i]->is_contiguous();
          ck = ck && epilogues_[i]->scalar_type() == kHalf;
          if (!ck)
            return *this;
        } break;
        default: {
        } break;
      }
    }
    fallback_ = false;
    return *this;
  }

  using DispatchResult = std::pair<std::function<void()>, bool>;

  template <int begin, int end, typename T>
  struct TupleExtractor {
    using type = std::tuple<T>;
  };

  template <int begin, typename FirstConfig, typename... OtherConfigs>
  struct TupleExtractor<
      begin,
      begin,
      std::tuple<FirstConfig, OtherConfigs...>> {
    using type = std::tuple<FirstConfig>;
  };

  template <int begin, int end, typename FirstConfig, typename... OtherConfigs>
  struct TupleExtractor<begin, end, std::tuple<FirstConfig, OtherConfigs...>> {
    template <typename AppendConfig, typename T>
    struct TupleAppender {
      using type = std::tuple<AppendConfig>;
    };

    template <typename AppendConfig, typename... RemainedConfigs>
    struct TupleAppender<AppendConfig, std::tuple<RemainedConfigs...>> {
      using type = std::tuple<AppendConfig, RemainedConfigs...>;
    };

    using type = typename TupleAppender<
        std::tuple_element_t<begin, std::tuple<FirstConfig, OtherConfigs...>>,
        typename TupleExtractor<
            begin + 1,
            end,
            std::tuple<FirstConfig, OtherConfigs...>>::type>::type;
  };

  template <typename scalar_t, typename ConfigsTuple>
  std::function<void()> binary_search(sycl::queue& q) {
    static constexpr int configs_size = std::tuple_size<ConfigsTuple>::value;
    if constexpr (configs_size == 1) {
      auto execute_function = [&]() {
        using Config = std::tuple_element_t<0, ConfigsTuple>;
        static constexpr int wg_m = Config::wg_m;
        static constexpr int wg_n = Config::wg_n;
        static constexpr int sg_m = Config::sg_m;
        static constexpr int sg_n = Config::sg_n;
        static constexpr int sg_k = Config::sg_k;
        static constexpr int gz = Config::gz;
        static constexpr int slm_ks = Config::slm_ks;
        static constexpr int arch = Config::arch;
        HGEMM_INT4_COMMON_DISPATCH(wg_m, wg_n, sg_m, sg_n, sg_k, gz, slm_ks, arch);
      };
      return execute_function;
    } else {
      static constexpr int mid = (configs_size - 1) / 2;
      using MiddleConfig = std::tuple_element_t<mid, ConfigsTuple>;
      if (MiddleConfig::less_than(m_, n_, k_, calib_gz_)) {
        return binary_search<
            scalar_t,
            typename TupleExtractor<mid + 1, configs_size - 1, ConfigsTuple>::
                type>(q);
      } else {
        return binary_search<
            scalar_t,
            typename TupleExtractor<0, mid, ConfigsTuple>::type>(q);
      }
    }
  }
  template <typename scalar_t, typename... configs>
  void dispatch(sycl::queue& q) {
    using ConfigsTuple = std::tuple<configs...>;
    auto gemm_caller = binary_search<scalar_t, ConfigsTuple>(q);
    gemm_caller();
  }

  void run() {
    using scalar_t =
        decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
    auto& q = dpcppGetCurrentQueue();
    if (arch_ == 1) {
      dispatch<scalar_t, ORDERED_GEMM_WINT4_CONFIG_SET_PVC>(q);
    } else {
      dispatch<scalar_t, ORDERED_GEMM_WINT4_CONFIG_SET_ARC>(q);
    }
    
  }
};
