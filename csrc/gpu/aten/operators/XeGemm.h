#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/ComputeEngine.h"
#include "xetla/hgemm.h"
#include "xetla/kernels/GEMM/hgemm_policy.h"

using namespace torch_ipex::xpu::xetla;

#define RECORD_FUNCTION_IMPL(F, M, N, K)            \
  char str__[100];                                  \
  sprintf(str__, "%s(%d, %d, %d)", "" #F, M, N, K); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

class HGEMM_XETLA final {
 public:
  enum EpilogueType {
    BIAS = 0,
    RES_ADD,
    RELU,
    GELU,
    RES_MUL,
    SILU,
  };

 private:
  enum {
    MAX_EPILOGUES = 4,
  };
  c10::MaybeOwned<Tensor> c_, a_, b_;
  SmallVector<c10::MaybeOwned<Tensor>, MAX_EPILOGUES> epilogue_tensors_;
  EpilogueType epilogue_types_[MAX_EPILOGUES];
  float epilogue_params_[MAX_EPILOGUES];
  int num_epilogues_ = 0;
  bool is_a_row_major_;
  bool is_a_col_major_;
  bool is_b_row_major_;
  bool is_b_col_major_;
  bool valid_ = false;
  int m_, n_, k_;
  float alpha_ = 1.0f;

  // template <uint32_t a, uint32_t b>
  // struct gcd {
  //   static constexpr uint32_t value = gcd<b, a % b>::value;
  // };
  // /// @brief
  // ///
  // /// @tparam a
  // template <uint32_t a>
  // struct gcd<a, 0> {
  //   static constexpr uint32_t value = a;
  // };

  static size_t get_acc_size(uint32_t matrix_m, uint32_t matrix_n) {
    return matrix_m * matrix_n;
  };

  static uint32_t find_ks_coop_num_y(uint32_t slm_kslicing, uint32_t sg_m) {
    uint32_t ks_coop_num_y = sg_m;
    while (slm_kslicing % sg_m != 0) {
      ks_coop_num_y = slm_kslicing % sg_m;
      slm_kslicing = sg_m;
      sg_m = ks_coop_num_y;
    }
    return ks_coop_num_y;
  }

  // template <
  //     uint32_t wg_m,
  //     uint32_t wg_n,
  //     uint32_t sg_n,
  //     uint32_t sg_m,
  //     uint32_t slm_kslicing>
  static size_t get_cnt_size(
      uint32_t matrix_m,
      uint32_t matrix_n,
      uint32_t wg_m,
      uint32_t wg_n,
      uint32_t sg_m,
      uint32_t sg_n,
      uint32_t slm_kslicing) {
    // return matrix_m * matrix_n;
    size_t group_range_m = (matrix_m + wg_m - 1) / wg_m;
    size_t group_range_n = (matrix_n + wg_n - 1) / wg_n;

    uint32_t wg_size_x = (wg_m + sg_m - 1) / sg_m;
    uint32_t wg_size_y = (wg_n + sg_n - 1) / sg_n;
    // uint32_t ks_coop_num_y = gcd<slm_kslicing, sg_m>::value;
    uint32_t ks_coop_num_y = find_ks_coop_num_y(slm_kslicing, sg_m);
    uint32_t coop_remain_num_x = slm_kslicing / ks_coop_num_y;
    bool has_redundant_wg = (coop_remain_num_x * 16) > sg_n;
    uint32_t tile_size_y = sg_m / ks_coop_num_y;
    uint32_t tile_size_x = has_redundant_wg ? 16 : sg_n / coop_remain_num_x;
    uint32_t ks_coop_num_x = sg_n / tile_size_x;

    uint32_t counter_size = 8;
    return group_range_m * group_range_n * wg_size_x * wg_size_y *
        ks_coop_num_y * ks_coop_num_x * counter_size;
  }

  void get_acc_and_cnt_tensor(
      uint32_t m_,
      uint32_t n_,
      uint32_t k_,
      bool is_b_row_major_,
      Tensor& acc_tensor_,
      Tensor& cnt_tensor_) {
    int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
    if (policy_id == -1) {
      acc_tensor_ = at::AtenIpexTypeXPU::empty(
          {0}, a_->options().dtype(at::kFloat), c10::nullopt);
      cnt_tensor_ = at::AtenIpexTypeXPU::empty(
          {0}, a_->options().dtype(at::kByte), c10::nullopt);
      return;
    }

    auto policy_config = hgemm_policy_traits[policy_id];
    uint32_t wg_m = policy_config.wg_m_;
    uint32_t wg_n = policy_config.wg_n_;
    uint32_t sg_m = policy_config.sg_m_;
    uint32_t sg_n = policy_config.sg_n_;
    uint32_t slm_ks = policy_config.slm_ks_;
    size_t acc_size = get_acc_size(m_, n_);
    size_t cnt_size = get_cnt_size(m_, n_, wg_m, wg_n, sg_m, sg_n, slm_ks);

    acc_tensor_ = at::AtenIpexTypeXPU::empty(
        {acc_size}, a_->options().dtype(at::kFloat), c10::nullopt);
    cnt_tensor_ = at::AtenIpexTypeXPU::empty(
        {cnt_size}, a_->options().dtype(at::kByte), c10::nullopt);
    return;
  }

 public:
  HGEMM_XETLA() = default;
  bool valid() const {
    return valid_;
  }
  HGEMM_XETLA& add_alpha(const float alpha) {
    alpha_ = alpha;
    return *this;
  }
  HGEMM_XETLA& add_matrix_c(const Tensor& c) {
    c_ = c10::MaybeOwned<Tensor>::borrowed(c);
    return *this;
  }
  HGEMM_XETLA& add_matrix_a(const Tensor& a) {
    a_ = c10::MaybeOwned<Tensor>::borrowed(a);
    return *this;
  }
  HGEMM_XETLA& add_matrix_b(const Tensor& b) {
    b_ = c10::MaybeOwned<Tensor>::borrowed(b);
    return *this;
  }
  HGEMM_XETLA& add_epilogue(
      const Tensor& t,
      const EpilogueType eptype,
      const float x = 1.0) {
    epilogue_tensors_.push_back(c10::MaybeOwned<Tensor>::borrowed(t));
    epilogue_params_[num_epilogues_] = x;
    epilogue_types_[num_epilogues_++] = eptype;
    return *this;
  }
  HGEMM_XETLA& build() {
#define __CHECK(X) \
  if (!(X))        \
    return *this;
    __CHECK(
        a_->scalar_type() == kHalf && b_->scalar_type() == kHalf &&
        c_->scalar_type() == kHalf);

    using scalar_t =
        decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
    __CHECK(reinterpret_cast<uint64_t>(c_->data_ptr<scalar_t>()) % 8 == 0);
    __CHECK(reinterpret_cast<uint64_t>(a_->data_ptr<scalar_t>()) % 8 == 0);
    __CHECK(reinterpret_cast<uint64_t>(b_->data_ptr<scalar_t>()) % 8 == 0);
    auto a_for_gemm = a_->flatten(0, -2);
    auto c_for_gemm = c_->flatten(0, -2);
    auto a_sizes = a_for_gemm.sizes();
    auto b_sizes = b_->sizes();
    auto c_sizes = c_for_gemm.sizes();
    m_ = a_sizes[0];
    k_ = a_sizes[1];
    n_ = b_sizes[0];
    __CHECK(k_ % 4 == 0 && n_ % 4 == 0);
    __CHECK((m_ == 1028) || (!(m_ > 1000 && n_ > 8192)));
    __CHECK(
        b_->dim() == 2 && b_sizes[0] == k_ && c_sizes[0] == m_ &&
        c_sizes[1] == n_);

    is_a_row_major_ = a_->is_contiguous();
    is_a_col_major_ = a_->transpose(0, 1).is_contiguous();
    is_b_row_major_ = b_->is_contiguous();
    is_b_col_major_ = b_->transpose(0, 1).is_contiguous();
    __CHECK(is_a_row_major_);
    __CHECK(is_b_row_major_);

    for (int i = 0; i < num_epilogues_; i++) {
      auto eptensor = epilogue_tensors_[i];
      switch (epilogue_types_[i]) {
        case BIAS: {
          __CHECK(
              eptensor->is_contiguous() && eptensor->scalar_type() == kHalf);
          __CHECK(eptensor->numel() == n_);
        } break;
        case RES_MUL:
        case RES_ADD: {
          __CHECK(
              eptensor->is_contiguous() && eptensor->scalar_type() == kHalf);
          auto eptensor_for_gemm = eptensor->flatten(0, -2);
          auto epsizes = eptensor_for_gemm.sizes();
          __CHECK(epsizes[0] == m_ && epsizes[1] == n_);
        } break;
        case RELU:
        case GELU:
        case SILU: {
        } break;
      }
    }
    valid_ = true;
    return *this;
#undef __CHECK
  }

  torch_ipex::xpu::xetla::GemmStatus run() {
    using scalar_t =
        decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
    auto& q = dpcppGetCurrentQueue();

    torch_ipex::xpu::xetla::GemmStatus status;

    Tensor acc_tensor_, cnt_tensor_;
    if (num_epilogues_ == 0) {
      RECORD_FUNCTION_IMPL(hgemm_common, m_, n_, k_)
      TORCH_CHECK(alpha_ == 1.0f);
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_common(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if (num_epilogues_ == 1 && epilogue_types_[0] == RES_ADD) {
      if (alpha_ == 1.0f) {
        RECORD_FUNCTION_IMPL(hgemm_res, m_, n_, k_)
        get_acc_and_cnt_tensor(
            m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
        int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
        if (policy_id < 0) {
          status = torch_ipex::xpu::xetla::GemmStatus::kError;
        } else {
          auto cgfs = hgemm_res(
              policy_id,
              reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
              reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
              reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
              reinterpret_cast<sycl::half*>(
                  epilogue_tensors_[0]->data_ptr<scalar_t>()),
              acc_tensor_.data_ptr<float>(),
              reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
              m_,
              n_,
              k_,
              epilogue_params_[0],
              is_b_row_major_);
          DPCPP_Q_SUBMIT_CGFS(q, cgfs);
          status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
        }
      } else {
        RECORD_FUNCTION_IMPL(hgemm_addmm, m_, n_, k_)
        get_acc_and_cnt_tensor(
            m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
        int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
        if (policy_id < 0) {
          status = torch_ipex::xpu::xetla::GemmStatus::kError;
        } else {
          auto cgfs = hgemm_addmm(
              policy_id,
              reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
              reinterpret_cast<sycl::half*>(
                  epilogue_tensors_[0]->data_ptr<scalar_t>()),
              reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
              reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
              acc_tensor_.data_ptr<float>(),
              reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
              m_,
              n_,
              k_,
              alpha_,
              epilogue_params_[0],
              is_b_row_major_);
          DPCPP_Q_SUBMIT_CGFS(q, cgfs);
          status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
        }
      }
    } else if (
        num_epilogues_ == 2 && epilogue_types_[0] == RES_ADD &&
        epilogue_types_[1] == RES_ADD) {
      RECORD_FUNCTION_IMPL(hgemm_res_res, m_, n_, k_)
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      TORCH_CHECK(alpha_ == 1.0f);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_res_res(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[0]->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[1]->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            epilogue_params_[0],
            epilogue_params_[1],
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if (num_epilogues_ == 1 && epilogue_types_[0] == BIAS) {
      RECORD_FUNCTION_IMPL(hgemm_bias, m_, n_, k_)
      TORCH_CHECK(alpha_ == 1.0f);
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_bias(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[0]->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            epilogue_params_[0],
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if (
        num_epilogues_ == 2 && epilogue_types_[0] == BIAS &&
        epilogue_types_[1] == RES_ADD) {
      RECORD_FUNCTION_IMPL(hgemm_bias_res, m_, n_, k_)
      TORCH_CHECK(alpha_ == 1.0f);
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_bias_res(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[0]->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[1]->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            epilogue_params_[0],
            epilogue_params_[1],
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if (
        num_epilogues_ == 3 && epilogue_types_[0] == BIAS &&
        epilogue_types_[1] == RES_ADD && epilogue_types_[2] == RES_ADD) {
      RECORD_FUNCTION_IMPL(hgemm_bias_res_res, m_, n_, k_)
      TORCH_CHECK(alpha_ == 1.0f);
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_bias_res_res(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[0]->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[1]->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[2]->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            epilogue_params_[0],
            epilogue_params_[1],
            epilogue_params_[2],
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if (
        num_epilogues_ == 2 && epilogue_types_[0] == BIAS &&
        epilogue_types_[1] == RELU) {
      RECORD_FUNCTION_IMPL(hgemm_bias_relu, m_, n_, k_)
      TORCH_CHECK(alpha_ == 1.0f);
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_bias_relu(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[0]->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            epilogue_params_[0],
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if (
        num_epilogues_ == 2 && epilogue_types_[0] == BIAS &&
        epilogue_types_[1] == GELU) {
      RECORD_FUNCTION_IMPL(hgemm_bias_gelu, m_, n_, k_)
      TORCH_CHECK(alpha_ == 1.0f);
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_bias_gelu(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[0]->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            epilogue_params_[0],
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if (num_epilogues_ == 1 && epilogue_types_[0] == RES_MUL) {
      RECORD_FUNCTION_IMPL(hgemm_resmul, m_, n_, k_)
      TORCH_CHECK(alpha_ == 1.0f);
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_resmul(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(
                epilogue_tensors_[0]->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if (num_epilogues_ == 1 && epilogue_types_[0] == SILU) {
      RECORD_FUNCTION_IMPL(hgemm_silu, m_, n_, k_)
      TORCH_CHECK(alpha_ == 1.0f);
      get_acc_and_cnt_tensor(
          m_, n_, k_, is_b_row_major_, acc_tensor_, cnt_tensor_);
      int policy_id = hgemm_find_policy_id(m_, n_, k_, is_b_row_major_);
      if (policy_id < 0) {
        status = torch_ipex::xpu::xetla::GemmStatus::kError;
      } else {
        auto cgfs = hgemm_silu(
            policy_id,
            reinterpret_cast<sycl::half*>(c_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(a_->data_ptr<scalar_t>()),
            reinterpret_cast<sycl::half*>(b_->data_ptr<scalar_t>()),
            acc_tensor_.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr()),
            m_,
            n_,
            k_,
            is_b_row_major_);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        status = torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else {
      TORCH_CHECK(false, "No mateched policy");
    }
    return status;
  };

  inline Tensor matmul_resize(const Tensor& a, const Tensor& output) {
    auto output_ = output.flatten(0, -2);
    int n = output_.sizes()[1];
    auto sizes = a.sym_sizes().vec();
    sizes[sizes.size() - 1] = n;
    return output.view_symint(sizes);
  }

  inline std::vector<std::tuple<int, int>> hgemm_split_m(
      const int m,
      const int n) {
    std::vector<std::tuple<int, int>> res;
    constexpr int slice_n = 4096;
    if (m > 4096 && n >= 4096) {
      for (int start_idx = 0; start_idx < m; start_idx += slice_n) {
        int remaining = m - start_idx;
        int len = slice_n < remaining ? slice_n : remaining;
        res.push_back(std::make_tuple(start_idx, len));
      }
    } else {
      res.push_back(std::make_tuple(0, m));
    }
    return res;
  }
};

inline Tensor matmul_resize(const Tensor& a, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = a.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

inline std::vector<std::tuple<int, int>> hgemm_split_m(
    const int m,
    const int n) {
  std::vector<std::tuple<int, int>> res;
  constexpr int slice_n = 4096;
  if (m > 4096 && n >= 4096) {
    for (int start_idx = 0; start_idx < m; start_idx += slice_n) {
      int remaining = m - start_idx;
      int len = slice_n < remaining ? slice_n : remaining;
      res.push_back(std::make_tuple(start_idx, len));
    }
  } else {
    res.push_back(std::make_tuple(0, m));
  }
  return res;
}

#undef RECORD_FUNCTION_IMPL
