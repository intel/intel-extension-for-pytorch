#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include <xetla/GEMM.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::xetla;

#define HGEMM_DISPATCH(F)                                               \
  {                                                                     \
    RECORD_FUNCTION("torch_ipex::" #F, c10::ArrayRef<c10::IValue>({})); \
    F(q,                                                                \
      reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),          \
      reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),          \
      reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),          \
      m_,                                                               \
      n_,                                                               \
      k_);                                                              \
  }

#define HGEMM_BIAS_DISPATCH(F)                                            \
  {                                                                       \
    RECORD_FUNCTION("torch_ipex::" #F, c10::ArrayRef<c10::IValue>({}));   \
    F(q,                                                                  \
      reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
      m_,                                                                 \
      n_,                                                                 \
      k_);                                                                \
  }

#define HGEMM_BIAS_RES_RES_DISPATCH(F)                                    \
  {                                                                       \
    RECORD_FUNCTION("torch_ipex::" #F, c10::ArrayRef<c10::IValue>({}));   \
    F(q,                                                                  \
      reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
      reinterpret_cast<sycl::half*>(epilogues_[1]->data_ptr<scalar_t>()), \
      reinterpret_cast<sycl::half*>(epilogues_[2]->data_ptr<scalar_t>()), \
      m_,                                                                 \
      n_,                                                                 \
      k_);                                                                \
  }

#define HGEMM_BIAS_GELU_DISPATCH(F)                                       \
  {                                                                       \
    RECORD_FUNCTION("torch_ipex::" #F, c10::ArrayRef<c10::IValue>({}));   \
    F(q,                                                                  \
      reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),            \
      reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
      m_,                                                                 \
      n_,                                                                 \
      k_);                                                                \
  }

#define HGEMM_COMMON_DISPATCH(F)                                      \
  {                                                                   \
    if (num_epilogues_ == 0)                                          \
      HGEMM_DISPATCH(hgemm##F)                                        \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == BIAS)        \
      HGEMM_BIAS_DISPATCH(hgemm_bias##F)                              \
    else if (                                                         \
        num_epilogues_ == 3 && epilogue_type_[0] == BIAS &&           \
        epilogue_type_[1] == RES_ADD && epilogue_type_[2] == RES_ADD) \
      HGEMM_BIAS_RES_RES_DISPATCH(hgemm_bias_res_res##F)              \
    else if (                                                         \
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&           \
        epilogue_type_[1] == GELU)                                    \
      HGEMM_BIAS_GELU_DISPATCH(hgemm_bias_gelu##F)                    \
  }

class HGEMMXetla final {
 public:
  enum EpilogueType {
    BIAS = 0,
    RES_ADD,
    GELU,
  };

 private:
  enum {
    MAX_EPILOGUES = 4,
  };
  Tensor *a_, *b_, *c_;
  Tensor* epilogues_[MAX_EPILOGUES];
  EpilogueType epilogue_type_[MAX_EPILOGUES];
  int num_epilogues_ = 0;
  bool is_a_row_major_;
  bool is_a_col_major_;
  bool is_b_row_major_;
  bool is_b_col_major_;
  bool fallback_;
  int m_, n_, k_;

 public:
  HGEMMXetla() = default;
  bool fallback() const {
    return fallback_;
  }
  HGEMMXetla& add_matrix_c(const Tensor& c) {
    c_ = const_cast<Tensor*>(&c);
    return *this;
  }
  HGEMMXetla& add_matrix_a(const Tensor& a) {
    a_ = const_cast<Tensor*>(&a);
    return *this;
  }
  HGEMMXetla& add_matrix_b(const Tensor& b) {
    b_ = const_cast<Tensor*>(&b);
    return *this;
  }
  HGEMMXetla& add_epilogue(const Tensor& t, EpilogueType eptype) {
    epilogues_[num_epilogues_] = const_cast<Tensor*>(&t);
    epilogue_type_[num_epilogues_++] = eptype;
    return *this;
  }

  HGEMMXetla& build() {
    fallback_ = true;
    if (a_->scalar_type() != kHalf || b_->scalar_type() != kHalf ||
        c_->scalar_type() != kHalf)
      return *this;
    if (!(a_->dim() == 2 && b_->dim() == 2 && c_->dim() == 2))
      return *this;
    is_a_row_major_ = a_->is_contiguous();
    is_a_col_major_ = a_->transpose(0, 1).is_contiguous();
    is_b_row_major_ = b_->is_contiguous();
    is_b_col_major_ = b_->transpose(0, 1).is_contiguous();
    auto a_sizes = a_->sizes();
    auto b_sizes = b_->sizes();
    auto c_sizes = c_->sizes();
    m_ = a_sizes[0];
    k_ = a_sizes[1];
    n_ = b_sizes[1];
    bool ck0 = b_sizes[0] == k_;
    bool ck1 = c_sizes[0] == m_ && c_sizes[1] == n_;
    bool ck2 = is_a_row_major_ && is_b_row_major_; // TODO:
    if (!(ck0 && ck1 && ck2))
      return *this;
    if (!(m_ <= 32 && n_ >= 4096 && k_ >= 4096)) // TODO:
      return *this;
    for (int i = 0; i < num_epilogues_; i++) {
      switch (epilogue_type_[i]) {
        case BIAS: {
          bool ck = epilogues_[i]->dim() == 1 && epilogues_[i]->is_contiguous();
          ck = ck && epilogues_[i]->sizes()[0] == n_;
          ck = ck && epilogues_[i]->scalar_type() == kHalf;
          if (!ck)
            return *this;
        } break;
        case RES_ADD: {
          bool ck = epilogues_[i]->dim() == 2;
          ck = ck && epilogues_[i]->sizes()[0] == m_ &&
              epilogues_[i]->sizes()[1] == n_;
          ck = ck && epilogues_[i]->is_contiguous();
          ck = ck && epilogues_[i]->scalar_type() == kHalf;
          if (!ck)
            return *this;
        } break;
        case GELU: {
        } break;
      }
    }
    fallback_ = false;
    return *this;
  }

  void run() {
    using scalar_t =
        decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
    auto& q = dpcppGetCurrentQueue();
    if (n_ >= 4096 && n_ < 16384 && k_ >= 4096 && k_ < 16384) {
      HGEMM_COMMON_DISPATCH(_32x64_8x16x16_2_);
      return;
    } else if (n_ >= 4096 && n_ < 16384 && k_ >= 16384) {
      HGEMM_COMMON_DISPATCH(_32x64_8x16x16_2_);
      return;
    } else {
      HGEMM_COMMON_DISPATCH(_8x512_8x16x16_1_);
      return;
    }
  }
};
