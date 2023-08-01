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

#define RECORD_FUNCTION_IMPL(F)                                             \
  char str__[100];                                                          \
  sprintf(str__, "%s_%d(%d, %d, %d)", "" #F, selected_policy_, m_, n_, k_); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

inline Tensor resize_as_mat1(const Tensor& mat1, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = mat1.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

class HGEMMXetla final {
 public:
  enum EpilogueType {
    BIAS = 0,
    RES_ADD,
    GELU,
    RES_MUL,
    SILU,
    SCALED_RES_ADD,
  };

 private:
  enum {
    MAX_EPILOGUES = 4,
  };
  Tensor *a_, *b_, *c_;
  Tensor* epilogues_[MAX_EPILOGUES];
  EpilogueType epilogue_type_[MAX_EPILOGUES];
  float pf32[MAX_EPILOGUES];
  int num_epilogues_ = 0;
  bool is_a_row_major_;
  bool is_a_col_major_;
  bool is_b_row_major_;
  bool is_b_col_major_;
  bool fallback_;
  int selected_policy_;
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
  HGEMMXetla& add_epilogue(
      const Tensor& t,
      EpilogueType eptype,
      const float x) {
    epilogues_[num_epilogues_] = const_cast<Tensor*>(&t);
    pf32[num_epilogues_] = x;
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
    bool ck2 = is_a_row_major_;
    if (!(ck0 && ck1 && ck2))
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
        case GELU:
        case SILU: {
        } break;
      }
    }
    fallback_ = false;
    selected_policy_ = select_gemm_config(
        m_, n_, k_, is_b_row_major_, 64); // 64 is subslice count per tile
    return *this;
  }

  void run() {
    using scalar_t =
        decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
    auto& q = dpcppGetCurrentQueue();

    if (num_epilogues_ == 0) {
      RECORD_FUNCTION_IMPL(hgemm)
      hgemm_policies[selected_policy_](
          q,
          reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
          m_,
          n_,
          k_);
    } else if (num_epilogues_ == 1 && epilogue_type_[0] == BIAS) {
      RECORD_FUNCTION_IMPL(hgemm_bias)
      hgemm_bias_policies[selected_policy_](
          q,
          reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()),
          m_,
          n_,
          k_);
    } else if (
        num_epilogues_ == 3 && epilogue_type_[0] == BIAS &&
        epilogue_type_[1] == RES_ADD && epilogue_type_[2] == RES_ADD) {
      RECORD_FUNCTION_IMPL(hgemm_bias_res_res)
      hgemm_bias_res_res_policies[selected_policy_](
          q,
          reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[1]->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[2]->data_ptr<scalar_t>()),
          m_,
          n_,
          k_);
    } else if (
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&
        epilogue_type_[1] == GELU) {
      RECORD_FUNCTION_IMPL(hgemm_bias_gelu)
      hgemm_bias_gelu_policies[selected_policy_](
          q,
          reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()),
          m_,
          n_,
          k_);
    } else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_MUL) {
      RECORD_FUNCTION_IMPL(hgemm_resmul)
      hgemm_resmul_policies[selected_policy_](
          q,
          reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()),
          m_,
          n_,
          k_);
    } else if (num_epilogues_ == 1 && epilogue_type_[0] == SILU) {
      RECORD_FUNCTION_IMPL(hgemm_silu)
      hgemm_silu_policies[selected_policy_](
          q,
          reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
          m_,
          n_,
          k_);
    } else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_ADD) {
      RECORD_FUNCTION_IMPL(hgemm_res)
      hgemm_res_policies[selected_policy_](
          q,
          reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()),
          m_,
          n_,
          k_);
    } else if (
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&
        epilogue_type_[1] == SCALED_RES_ADD) {
      RECORD_FUNCTION_IMPL(hgemm_bias_res)
      hgemm_bias_res_policies[selected_policy_](
          q,
          reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()),
          reinterpret_cast<sycl::half*>(epilogues_[1]->data_ptr<scalar_t>()),
          (scalar_t)pf32[1],
          m_,
          n_,
          k_);
    }
  }
};
