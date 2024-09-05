#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"
#include "xetla/GEMM_INT4.h"

using namespace torch_ipex::xpu::xetla;

#define RECORD_FUNCTION_IMPL(                                       \
    F,                                                              \
    WG_M,                                                           \
    WG_N,                                                           \
    SG_M,                                                           \
    SG_N,                                                           \
    SG_K,                                                           \
    GZ,                                                             \
    SLM_KS,                                                         \
    L3_KS,                                                          \
    SYNC_FREQ,                                                      \
    STAGES,                                                         \
    ARCH)                                                           \
  char str__[100];                                                  \
  sprintf(                                                          \
      str__,                                                        \
      "%s(%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)", \
      "" #F,                                                        \
      WG_M,                                                         \
      WG_N,                                                         \
      SG_M,                                                         \
      SG_N,                                                         \
      SG_K,                                                         \
      GZ,                                                           \
      SLM_KS,                                                       \
      L3_KS,                                                        \
      SYNC_FREQ,                                                    \
      STAGES,                                                       \
      ARCH,                                                         \
      m_,                                                           \
      n_,                                                           \
      k_);                                                          \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

#define HGEMM_INT4_DISPATCH(                                \
    F,                                                      \
    WG_M,                                                   \
    WG_N,                                                   \
    SG_M,                                                   \
    SG_N,                                                   \
    SG_K,                                                   \
    GZ,                                                     \
    SLM_KS,                                                 \
    L3_KS,                                                  \
    SYNC_FREQ,                                              \
    STAGES,                                                 \
    ARCH)                                                   \
  F<scalar_t,                                               \
    WG_M,                                                   \
    WG_N,                                                   \
    SG_M,                                                   \
    SG_N,                                                   \
    SG_K,                                                   \
    GZ,                                                     \
    SLM_KS,                                                 \
    L3_KS,                                                  \
    SYNC_FREQ,                                              \
    STAGES,                                                 \
    ARCH>(                                                  \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()), \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),      \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),     \
      weight_zp_ptr_,                                       \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                       \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()), \
      m_,                                                   \
      n_,                                                   \
      k_);

#define HGEMM_INT4_BIAS_DISPATCH(                             \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_BIAS_RES_RES_DISPATCH(                     \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      reinterpret_cast<scalar_t*>(epilogues_[1]->data_ptr()), \
      reinterpret_cast<scalar_t*>(epilogues_[2]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_BIAS_GELU_DISPATCH(                        \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_RES_DISPATCH(                              \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_RESMUL_DISPATCH(                           \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_MLP_SILU_MUL_DISPATCH(                   \
    F,                                                      \
    WG_M,                                                   \
    WG_N,                                                   \
    SG_M,                                                   \
    SG_N,                                                   \
    SG_K,                                                   \
    GZ,                                                     \
    SLM_KS,                                                 \
    L3_KS,                                                  \
    SYNC_FREQ,                                              \
    STAGES,                                                 \
    ARCH)                                                   \
  F<scalar_t,                                               \
    WG_M,                                                   \
    WG_N,                                                   \
    SG_M,                                                   \
    SG_N,                                                   \
    SG_K,                                                   \
    GZ,                                                     \
    SLM_KS,                                                 \
    L3_KS,                                                  \
    SYNC_FREQ,                                              \
    STAGES,                                                 \
    ARCH>(                                                  \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()), \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),      \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),     \
      weight_zp_ptr_,                                       \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                       \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()), \
      m_,                                                   \
      n_,                                                   \
      k_);

#define HGEMM_INT4_MLP_BIAS_SILU_MUL_DISPATCH(                \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);
#define HGEMM_INT4_MLP_SILU_MUL_BIAS_DISPATCH(                \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[2]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);
#define HGEMM_INT4_MLP_BIAS_SILU_MUL_BIAS_DISPATCH(           \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      reinterpret_cast<scalar_t*>(epilogues_[3]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_QKV_DISPATCH(                            \
    F,                                                      \
    WG_M,                                                   \
    WG_N,                                                   \
    SG_M,                                                   \
    SG_N,                                                   \
    SG_K,                                                   \
    GZ,                                                     \
    SLM_KS,                                                 \
    L3_KS,                                                  \
    SYNC_FREQ,                                              \
    STAGES,                                                 \
    ARCH)                                                   \
  F<scalar_t,                                               \
    WG_M,                                                   \
    WG_N,                                                   \
    SG_M,                                                   \
    SG_N,                                                   \
    SG_K,                                                   \
    GZ,                                                     \
    SLM_KS,                                                 \
    L3_KS,                                                  \
    SYNC_FREQ,                                              \
    STAGES,                                                 \
    ARCH>(                                                  \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()), \
      outputs_[0]->stride(0),                               \
      offset_n_k,                                           \
      reinterpret_cast<scalar_t*>(outputs_[1]->data_ptr()), \
      outputs_[1]->stride(0),                               \
      offset_n_v,                                           \
      reinterpret_cast<scalar_t*>(outputs_[2]->data_ptr()), \
      outputs_[2]->stride(0),                               \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),      \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),     \
      weight_zp_ptr_,                                       \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                       \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()), \
      m_,                                                   \
      n_,                                                   \
      k_);

#define HGEMM_INT4_SILU_MUL_DISPATCH(                         \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[1]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_BIAS_SILU_MUL_DISPATCH(                    \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      reinterpret_cast<scalar_t*>(epilogues_[2]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_BIAS_ADD_DISPATCH(                         \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      reinterpret_cast<scalar_t*>(epilogues_[1]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_QKV_BIAS_DISPATCH(                         \
    F,                                                        \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH)                                                     \
  F<scalar_t,                                                 \
    WG_M,                                                     \
    WG_N,                                                     \
    SG_M,                                                     \
    SG_N,                                                     \
    SG_K,                                                     \
    GZ,                                                       \
    SLM_KS,                                                   \
    L3_KS,                                                    \
    SYNC_FREQ,                                                \
    STAGES,                                                   \
    ARCH>(                                                    \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()),   \
      outputs_[0]->stride(0),                                 \
      offset_n_k,                                             \
      reinterpret_cast<scalar_t*>(outputs_[1]->data_ptr()),   \
      outputs_[1]->stride(0),                                 \
      offset_n_v,                                             \
      reinterpret_cast<scalar_t*>(outputs_[2]->data_ptr()),   \
      outputs_[2]->stride(0),                                 \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),        \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),       \
      weight_zp_ptr_,                                         \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()),   \
      reinterpret_cast<scalar_t*>(epilogues_[0]->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                         \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()),   \
      m_,                                                     \
      n_,                                                     \
      k_);

#define HGEMM_INT4_SILU_DISPATCH(                           \
    F,                                                      \
    WG_M,                                                   \
    WG_N,                                                   \
    SG_M,                                                   \
    SG_N,                                                   \
    SG_K,                                                   \
    GZ,                                                     \
    SLM_KS,                                                 \
    L3_KS,                                                  \
    SYNC_FREQ,                                              \
    STAGES,                                                 \
    ARCH)                                                   \
  F<scalar_t,                                               \
    WG_M,                                                   \
    WG_N,                                                   \
    SG_M,                                                   \
    SG_N,                                                   \
    SG_K,                                                   \
    GZ,                                                     \
    SLM_KS,                                                 \
    L3_KS,                                                  \
    SYNC_FREQ,                                              \
    STAGES,                                                 \
    ARCH>(                                                  \
      reinterpret_cast<scalar_t*>(outputs_[0]->data_ptr()), \
      reinterpret_cast<scalar_t*>(input_->data_ptr()),      \
      reinterpret_cast<uint32_t*>(weight_->data_ptr()),     \
      weight_zp_ptr_,                                       \
      reinterpret_cast<scalar_t*>(weight_scl_->data_ptr()), \
      acc_tensor_->data_ptr<float>(),                       \
      reinterpret_cast<uint32_t*>(cnt_tensor_->data_ptr()), \
      m_,                                                   \
      n_,                                                   \
      k_);

#define HGEMM_INT4_COMMON_DISPATCH_IMPL(DISPATCHER, F, ...) \
  {                                                         \
    RECORD_FUNCTION_IMPL(F, ##__VA_ARGS__)                  \
    auto cgfs = DISPATCHER(F, ##__VA_ARGS__);               \
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);                           \
  }

#define HGEMM_INT4_COMMON_DISPATCH(                                           \
    WG_M, WG_N, SG_M, SG_N, SG_K, GZ, SLM_KS, L3_KS, SYNC_FREQ, STAGES, ARCH) \
  {                                                                           \
    if (num_epilogues_ == 0)                                                  \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_DISPATCH,                                                \
          hgemm_wint4,                                                        \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == BIAS)                \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_BIAS_DISPATCH,                                           \
          hgemm_bias_wint4,                                                   \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (                                                                 \
        num_epilogues_ == 3 && epilogue_type_[0] == BIAS &&                   \
        epilogue_type_[1] == RES_ADD && epilogue_type_[2] == RES_ADD)         \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_BIAS_RES_RES_DISPATCH,                                   \
          hgemm_bias_res_res_wint4,                                           \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (                                                                 \
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&                   \
        epilogue_type_[1] == GELU)                                            \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_BIAS_GELU_DISPATCH,                                      \
          hgemm_bias_gelu_wint4,                                              \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_ADD)             \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_RES_DISPATCH,                                            \
          hgemm_res_wint4,                                                    \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_MUL)             \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_RESMUL_DISPATCH,                                         \
          hgemm_mul_wint4,                                                    \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == SPLIT3)              \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_QKV_DISPATCH,                                            \
          hgemm_qkv_wint4,                                                    \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == SILU)                \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_SILU_DISPATCH,                                           \
          hgemm_silu_wint4,                                                   \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (                                                                 \
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&                   \
        epilogue_type_[1] == SPLIT3)                                          \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_QKV_BIAS_DISPATCH,                                       \
          hgemm_qkv_bias_wint4,                                               \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (                                                                 \
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&                   \
        epilogue_type_[1] == RES_ADD)                                         \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_BIAS_ADD_DISPATCH,                                       \
          hgemm_bias_add_wint4,                                               \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (                                                                 \
        num_epilogues_ == 3 && epilogue_type_[0] == BIAS &&                   \
        epilogue_type_[1] == SILU && epilogue_type_[2] == RES_MUL)            \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_BIAS_SILU_MUL_DISPATCH,                                  \
          hgemm_bias_silu_mul_wint4,                                          \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (                                                                 \
        num_epilogues_ == 2 && epilogue_type_[0] == SILU &&                   \
        epilogue_type_[1] == RES_MUL)                                         \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                        \
          HGEMM_INT4_SILU_MUL_DISPATCH,                                       \
          hgemm_silu_mul_wint4,                                               \
          WG_M,                                                               \
          WG_N,                                                               \
          SG_M,                                                               \
          SG_N,                                                               \
          SG_K,                                                               \
          GZ,                                                                 \
          SLM_KS,                                                             \
          L3_KS,                                                              \
          SYNC_FREQ,                                                          \
          STAGES,                                                             \
          ARCH)                                                               \
    else if (std::any_of(                                                     \
                 epilogue_type_,                                              \
                 epilogue_type_ + num_epilogues_,                             \
                 [](EpilogueType epi_type) {                                  \
                   return epi_type == GATE_UP_MUL;                            \
                 })) {                                                        \
      if (num_epilogues_ == 2 && epilogue_type_[0] == SILU &&                 \
          epilogue_type_[1] == GATE_UP_MUL)                                   \
        HGEMM_INT4_COMMON_DISPATCH_IMPL(                                      \
            HGEMM_INT4_MLP_SILU_MUL_DISPATCH,                                 \
            hgemm_mlp_silu_mul_wint4,                                         \
            WG_M,                                                             \
            WG_N,                                                             \
            SG_M,                                                             \
            SG_N,                                                             \
            SG_K,                                                             \
            GZ,                                                               \
            SLM_KS,                                                           \
            L3_KS,                                                            \
            SYNC_FREQ,                                                        \
            STAGES,                                                           \
            ARCH)                                                             \
      else if (                                                               \
          num_epilogues_ == 3 && epilogue_type_[0] == BIAS &&                 \
          epilogue_type_[1] == SILU && epilogue_type_[2] == GATE_UP_MUL)      \
        HGEMM_INT4_COMMON_DISPATCH_IMPL(                                      \
            HGEMM_INT4_MLP_BIAS_SILU_MUL_DISPATCH,                            \
            hgemm_mlp_bias_silu_mul_wint4,                                    \
            WG_M,                                                             \
            WG_N,                                                             \
            SG_M,                                                             \
            SG_N,                                                             \
            SG_K,                                                             \
            GZ,                                                               \
            SLM_KS,                                                           \
            L3_KS,                                                            \
            SYNC_FREQ,                                                        \
            STAGES,                                                           \
            ARCH)                                                             \
      else if (                                                               \
          num_epilogues_ == 3 && epilogue_type_[0] == SILU &&                 \
          epilogue_type_[1] == GATE_UP_MUL && epilogue_type_[2] == BIAS)      \
        HGEMM_INT4_COMMON_DISPATCH_IMPL(                                      \
            HGEMM_INT4_MLP_SILU_MUL_BIAS_DISPATCH,                            \
            hgemm_mlp_silu_mul_bias_wint4,                                    \
            WG_M,                                                             \
            WG_N,                                                             \
            SG_M,                                                             \
            SG_N,                                                             \
            SG_K,                                                             \
            GZ,                                                               \
            SLM_KS,                                                           \
            L3_KS,                                                            \
            SYNC_FREQ,                                                        \
            STAGES,                                                           \
            ARCH)                                                             \
      else if (                                                               \
          num_epilogues_ == 4 && epilogue_type_[0] == BIAS &&                 \
          epilogue_type_[1] == SILU && epilogue_type_[2] == GATE_UP_MUL &&    \
          epilogue_type_[3] == BIAS)                                          \
        HGEMM_INT4_COMMON_DISPATCH_IMPL(                                      \
            HGEMM_INT4_MLP_BIAS_SILU_MUL_BIAS_DISPATCH,                       \
            hgemm_mlp_bias_silu_mul_bias_wint4,                               \
            WG_M,                                                             \
            WG_N,                                                             \
            SG_M,                                                             \
            SG_N,                                                             \
            SG_K,                                                             \
            GZ,                                                               \
            SLM_KS,                                                           \
            L3_KS,                                                            \
            SYNC_FREQ,                                                        \
            STAGES,                                                           \
            ARCH)                                                             \
      else                                                                    \
        TORCH_CHECK(false, "Unsupported gate_up epilogue type");              \
    } else                                                                    \
      TORCH_CHECK(false, "Unsupported epilogue type");                        \
  }

struct GemmWint4Config {
  int wg_m;
  int wg_n;
  int sg_m;
  int sg_n;
  int sg_k;
  int slm_ks;
  int max_m;
  int max_n;
  int max_k;
  int l3_ks;
  int sync_freq;
  int stages;
  bool fulfill(int m, int n, int k) const {
    return max_m >= m && max_n >= n && max_k >= k && k % sg_k == 0;
  }
};

#define MAX_INT std::numeric_limits<int>::max()

// GEMM_INT4 will be executed with the first config that can fulfill the input
template <gpu::xetla::gpu_arch arch_tag>
static inline constexpr std::array<GemmWint4Config, 0> ordered_config_set{};

// clang-format off
template <>
static inline constexpr std::array ordered_config_set<gpu::xetla::gpu_arch::XeHpc>{
  GemmWint4Config{1, 1, 1, 1, 512, 1, MAX_INT, MAX_INT, MAX_INT, 1, 0, 0},
  GemmWint4Config{1, 1, 1, 1, 256, 1, MAX_INT, MAX_INT, MAX_INT, 1, 0, 0},
  GemmWint4Config{1, 1, 1, 1, 128, 1, MAX_INT, MAX_INT, MAX_INT, 1, 0, 0},
};
template <>
static inline constexpr std::array ordered_config_set<gpu::xetla::gpu_arch::XeHpg>{
  GemmWint4Config{1, 1, 1, 1, 512, 1, MAX_INT, MAX_INT, MAX_INT, 1, 0, 0},
  GemmWint4Config{1, 1, 1, 1, 256, 1, MAX_INT, MAX_INT, MAX_INT, 1, 0, 0},
  GemmWint4Config{1, 1, 1, 1, 128, 1, MAX_INT, MAX_INT, MAX_INT, 1, 0, 0},
};
template <>
static inline constexpr std::array ordered_config_set<gpu::xetla::gpu_arch::XeLpg>{
  GemmWint4Config{1, 1, 1, 1, 512, 1, 8,       MAX_INT, MAX_INT, 1, 0, 0},
  GemmWint4Config{1, 1, 1, 1, 256, 1, 8,       MAX_INT, MAX_INT, 1, 0, 0},
  GemmWint4Config{1, 1, 1, 1, 128, 1, 8,       MAX_INT, MAX_INT, 1, 0, 0},
  GemmWint4Config{4, 1, 4, 1, 128, 1, MAX_INT, MAX_INT, MAX_INT, 1, 0, 0},
};
// clang-format on

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
    GATE_UP_MUL,
  };

 private:
  enum {
    MAX_EPILOGUES = 4,
  };
  Tensor *input_ = nullptr, *weight_ = nullptr, *weight_scl_ = nullptr,
         *acc_tensor_ = nullptr, *cnt_tensor_ = nullptr;
  uint32_t* weight_zp_ptr_ = nullptr;
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
  int offset_n_k = 0, offset_n_v = 0; // n-dim offset, for qkv fusion
  int64_t group_size_;
  int8_t arch_ = static_cast<int>(gpu::xetla::gpu_arch::XeHpc);
  template <uint32_t a, uint32_t b>
  struct gcd {
    static constexpr uint32_t value = gcd<b, a % b>::value;
  };
  /// @brief
  ///
  /// @tparam a
  template <uint32_t a>
  struct gcd<a, 0> {
    static constexpr uint32_t value = a;
  };

  static size_t get_acc_size(uint32_t matrix_m, uint32_t matrix_n) {
    return matrix_m * matrix_n;
  };

  template <
      uint32_t wg_m,
      uint32_t wg_n,
      uint32_t sg_m,
      uint32_t sg_n,
      uint32_t slm_kslicing>
  static size_t get_cnt_size(uint32_t matrix_m, uint32_t matrix_n) {
    size_t group_range_m = (matrix_m + wg_m - 1) / wg_m;
    size_t group_range_n = (matrix_n + wg_n - 1) / wg_n;

    static constexpr uint32_t wg_size_x = (wg_m + sg_m - 1) / sg_m;
    static constexpr uint32_t wg_size_y = (wg_n + sg_n - 1) / sg_n;
    static constexpr uint32_t ks_coop_num_y = gcd<slm_kslicing, sg_m>::value;
    static constexpr uint32_t coop_remain_num_x = slm_kslicing / ks_coop_num_y;
    static constexpr bool has_redundant_wg = (coop_remain_num_x * 16) > sg_n;
    static constexpr uint32_t tile_size_y = sg_m / ks_coop_num_y;
    static constexpr uint32_t tile_size_x =
        has_redundant_wg ? 16 : sg_n / coop_remain_num_x;
    static constexpr uint32_t ks_coop_num_x = sg_n / tile_size_x;

    static constexpr uint32_t counter_size = 8;
    return group_range_m * group_range_n * wg_size_x * wg_size_y *
        ks_coop_num_y * ks_coop_num_x * counter_size;
  };

 public:
  HGEMMXetla_INT4() = default;
  bool fallback() const {
    return fallback_;
  }
  HGEMMXetla_INT4& add_arch(int8_t arch) {
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
    Tensor* weight_zp_ = const_cast<Tensor*>(&zero_points);
    if (weight_zp_->defined())
      weight_zp_ptr_ = reinterpret_cast<uint32_t*>(weight_zp_->data_ptr());
    return *this;
  }
  HGEMMXetla_INT4& add_group_size(int64_t group_size) {
    group_size_ = group_size;
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
    const auto input_dtype = input_->scalar_type();
    if (input_dtype != kHalf && input_dtype != kBFloat16) {
      std::cout << "input dtype check fail: " << input_dtype << std::endl;
      return *this;
    }
    if (weight_->scalar_type() != kByte && weight_->scalar_type() != kQUInt8 &&
        weight_->scalar_type() != kChar && weight_->scalar_type() != kInt) {
      std::cout << "weight dtype check fail: " << weight_->scalar_type()
                << std::endl;
      return *this;
    }
    if (std::any_of(outputs_.begin(), outputs_.end(), [input_dtype](Tensor* o) {
          return o->scalar_type() != input_dtype;
        })) {
      std::cout << "outputs dtype check fail!" << std::endl;
      return *this;
    }
    bool has_split3 =
        (epilogue_type_[0] == SPLIT3 ||
         (epilogue_type_[0] == BIAS && epilogue_type_[1] == SPLIT3));
    bool has_gate_up = std::any_of(
        epilogue_type_,
        epilogue_type_ + num_epilogues_,
        [](EpilogueType epi_type) { return epi_type == GATE_UP_MUL; });
    if (input_->dim() != 2) {
      std::cout << "input dim check fail!" << std::endl;
      return *this;
    }
    if (std::any_of(outputs_.begin(), outputs_.end(), [](Tensor* out) {
          return out->dim() != 2;
        })) {
      std::cout << "output dim check fail!" << std::endl;
      return *this;
    }
    if ((has_gate_up) ? weight_->dim() != 3 : weight_->dim() != 2) {
      std::cout << "weight dim check fail!" << std::endl;
      return *this;
    }
    is_a_row_major_ = input_->is_contiguous();
    is_a_col_major_ = input_->transpose(0, 1).is_contiguous();
    is_b_row_major_ = weight_->is_contiguous();
    is_b_col_major_ = weight_->transpose(0, 1).is_contiguous();
    auto a_sizes = input_->sizes();
    auto b_sizes = weight_->sizes();
    m_ = a_sizes[0];
    k_ = a_sizes[1];
    // Normalize calibration group size.
    if (group_size_ == -1 || group_size_ == k_)
      group_size_ = 0;
    // Set correct n dim.
    n_ = b_sizes[has_gate_up ? 1 : 0];

    if (has_split3) {
      offset_n_k = outputs_[0]->sizes()[1];
      offset_n_v = offset_n_k + outputs_[1]->sizes()[1];
      auto offset_end = offset_n_v + outputs_[2]->sizes()[1];
      if (offset_end != n_) {
        std::cout << "output size for the k/v tensor do not match output size!"
                  << std::endl;
        return *this;
      }
    }
    for (int i = 0; i < num_epilogues_; i++) {
      switch (epilogue_type_[i]) {
        case BIAS: {
          if (epilogues_[i]->dim() != 1) {
            std::cout << "bias dim check fail!" << std::endl;
            return *this;
          }
          if (!epilogues_[i]->is_contiguous()) {
            std::cout << "bias contiguity check fail!" << std::endl;
            return *this;
          }
          if (epilogues_[i]->sizes()[0] != n_) {
            std::cout << "bias size check fail!" << std::endl;
            return *this;
          }
          if (epilogues_[i]->scalar_type() != input_dtype) {
            std::cout << "bias dtype check fail!" << std::endl;
            return *this;
          }
        } break;
        case RES_MUL:
        case RES_ADD: {
          bool ck = epilogues_[i]->dim() == 2;
          ck = ck && epilogues_[i]->sizes()[0] == m_ &&
              epilogues_[i]->sizes()[1] == n_;
          ck = ck && epilogues_[i]->is_contiguous();
          ck = ck && epilogues_[i]->scalar_type() == input_dtype;
          if (!ck) {
            std::cout << "res_add ck check fail!" << std::endl;
            return *this;
          }
        } break;
        default: {
        } break;
      }
    }
    fallback_ = false;
    return *this;
  }

  using DispatchResult = std::pair<std::function<void()>, bool>;
  using gpu_arch = gpu::xetla::gpu_arch;

  template <typename scalar_t, gpu_arch arch_tag, int gz, int idx = 0>
  void search_and_run(sycl::queue& q) {
    static constexpr auto ConfigsTuple = ordered_config_set<arch_tag>;
    if constexpr (idx >= ConfigsTuple.size()) {
      TORCH_CHECK(false, "No available config for current shape!")
      return;
    } else if (ConfigsTuple[idx].fulfill(m_, n_, k_)) {
      static constexpr auto CurrConfig = ConfigsTuple[idx];
      static constexpr int wg_m = CurrConfig.wg_m;
      static constexpr int wg_n = CurrConfig.wg_n;
      static constexpr int sg_m = CurrConfig.sg_m;
      static constexpr int sg_n = CurrConfig.sg_n;
      static constexpr int sg_k = CurrConfig.sg_k;
      // static constexpr int gz = gz;
      static constexpr int slm_ks = CurrConfig.slm_ks;
      static constexpr int arch = static_cast<int>(arch_tag);
      static constexpr int l3_ks = CurrConfig.l3_ks;
      static constexpr int sync_freq = CurrConfig.sync_freq;
      static constexpr int stages = CurrConfig.stages;
      // allocate temp buffers for global split
      Tensor acc_tensor, cnt_tensor;
      if constexpr (l3_ks > 1) {
        size_t acc_size = get_acc_size(m_, n_);
        size_t cnt_size = get_cnt_size<wg_m, wg_n, sg_m, sg_n, slm_ks>(m_, n_);
        if (epilogue_type_[0] == GATE_UP_MUL)
          acc_size *= 2;
        acc_tensor = at::AtenIpexTypeXPU::empty(
            {acc_size}, input_->options().dtype(at::kFloat), c10::nullopt);
        acc_tensor = at::AtenIpexTypeXPU::empty(
            {cnt_size}, input_->options().dtype(at::kByte), c10::nullopt);
        acc_tensor_ = const_cast<Tensor*>(&acc_tensor);
        cnt_tensor_ = const_cast<Tensor*>(&cnt_tensor);
      } else {
        static auto null_tensor_float = at::AtenIpexTypeXPU::empty(
            {0}, input_->options().dtype(at::kFloat), c10::nullopt);
        static auto null_tensor_byte = at::AtenIpexTypeXPU::empty(
            {0}, input_->options().dtype(at::kByte), c10::nullopt);
        acc_tensor_ = const_cast<Tensor*>(&null_tensor_float);
        cnt_tensor_ = const_cast<Tensor*>(&null_tensor_byte);
      }
      HGEMM_INT4_COMMON_DISPATCH(
          wg_m,
          wg_n,
          sg_m,
          sg_n,
          sg_k,
          gz,
          slm_ks,
          l3_ks,
          sync_freq,
          stages,
          arch);
    } else {
      search_and_run<scalar_t, arch_tag, gz, idx + 1>(q);
    }
  }
  template <typename scalar_t, gpu_arch arch_tag, int idx = 0>
  void dispatch(sycl::queue& q) {
    static constexpr std::array supported_gzs = {
        0, 32, 64, 128, 256, 512, 1024};
    if constexpr (idx >= supported_gzs.size()) {
      TORCH_CHECK(false, "No available implementation for current gz!")
    } else if (group_size_ == supported_gzs[idx]) {
      search_and_run<scalar_t, arch_tag, supported_gzs[idx]>(q);
    } else {
      dispatch<scalar_t, arch_tag, idx + 1>(q);
    }
  }
  template <gpu_arch arch_tag>
  void dispatch(sycl::queue& q) {
    switch (input_->scalar_type()) {
      case kHalf:
        return dispatch<sycl::half, arch_tag>(q);
      case kBFloat16:
        if constexpr (arch_tag == gpu_arch::XeLpg) {
          TORCH_CHECK(false, "XeLpg does not support bfloat16!")
        } else {
#if __INTEL_LLVM_COMPILER >= 20240200
          return dispatch<sycl::ext::oneapi::bfloat16, arch_tag>(q);
#else
          TORCH_CHECK(
              false, "bfloat16 is only supported on oneAPI 2024.2 and above!")
#endif
        }
      default:
        TORCH_CHECK(false, "Unexpected input dtype!");
        break;
    }
  }

  void run() {
    auto& q = dpcppGetCurrentQueue();
#ifdef USE_XETLA_XE_HPC
    if (arch_ == static_cast<int>(gpu_arch::XeHpc)) {
      dispatch<gpu_arch::XeHpc>(q);
      return;
    }
#endif
#ifdef USE_XETLA_XE_HPG
    if (arch_ == static_cast<int>(gpu_arch::XeHpg)) {
      dispatch<gpu_arch::XeHpg>(q);
      return;
    }
#endif
#ifdef USE_XETLA_XE_LPG
    if (arch_ == static_cast<int>(gpu_arch::XeLpg)) {
      dispatch<gpu_arch::XeLpg>(q);
      return;
    }
#endif
    TORCH_CHECK(false, "No available implementation for current architecture!");
  }
};
