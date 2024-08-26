#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"
#include "xetla/DEQUANTIZE_INT4.h"

using namespace torch_ipex::xpu::xetla;

class INT4_DEQUANTIZE_XETLA final {
 public:
  enum WeightCompressType {
    int4x2 = 0,
    int4x8,
  };

 private:
  Tensor *dequant_weight_ = nullptr, *weight_ = nullptr, *weight_scl_ = nullptr,
         *weight_zp_ptr_ = nullptr;
  int n_, k_;
  int64_t group_size_;
  WeightCompressType compress_type_;

 public:
  INT4_DEQUANTIZE_XETLA() = default;
  INT4_DEQUANTIZE_XETLA& add_dequant_weight(const Tensor& dequant_weight) {
    dequant_weight_ = const_cast<Tensor*>(&dequant_weight);
    return *this;
  }
  INT4_DEQUANTIZE_XETLA& add_weight(const Tensor& weight) {
    weight_ = const_cast<Tensor*>(&weight);
    return *this;
  }
  INT4_DEQUANTIZE_XETLA& add_scl(const Tensor& weight_scl) {
    weight_scl_ = const_cast<Tensor*>(&weight_scl);
    return *this;
  }
  INT4_DEQUANTIZE_XETLA& add_zp(const Tensor& weight_zp) {
    weight_zp_ptr_ = const_cast<Tensor*>(&weight_zp);
    return *this;
  }
  INT4_DEQUANTIZE_XETLA& add_group_size(int64_t group_size) {
    group_size_ = group_size;
    return *this;
  }
  INT4_DEQUANTIZE_XETLA& add_weight_compress_type(WeightCompressType type) {
    compress_type_ = type;
    return *this;
  }

  void int4x8_shape_check() {
    TORCH_CHECK(
        weight_scl_->sizes()[0] == n_ &&
            weight_scl_->sizes()[1] == k_ / group_size_,
        "int4 dequantize: invalid scale matrix size. only support col-major scale");
    // TODO(zhe): handle zp after support asym quantized weight.
  }

  INT4_DEQUANTIZE_XETLA& check() {
    n_ = dequant_weight_->sizes()[1];
    k_ = dequant_weight_->sizes()[0];
    TORCH_CHECK(
        weight_scl_->scalar_type() == at::kHalf,
        "int4 dequantize: scale dtype should be half type.");
    TORCH_CHECK(
        k_ % group_size_ == 0,
        "int4 dequantize: reduce-dim must be divisible by group size.");
    if (compress_type_ == WeightCompressType::int4x8) {
      int4x8_shape_check();
    }
    return *this;
  }

  struct Int4DequantizeConfig {
    int wg_n;
    int wg_k;
    int sg_n;
    int sg_k;
    int k_stride;
  };

  // LGP,HPG share this same config as benchmark result show that this config
  // can achieve 90%+ bandwidth utilization.
  static inline constexpr Int4DequantizeConfig kernel_config{
      16,
      128,
      16,
      128,
      32};

  static constexpr std::array supproted_gzs = {16, 32, 64, 128, 256, 512, 1024};

  template <int idx = 0>
  void dispatch(sycl::queue& q) {
    if constexpr (idx >= supproted_gzs.size()) {
      TORCH_CHECK(false, "No available implementation for current gz!");
    } else {
      if (group_size_ == supproted_gzs[idx]) {
        if (weight_zp_ptr_ == nullptr) {
          auto cfgs = {xetla_dequantize_int4_weight<
              sycl::half,
              gpu::xetla::quant_mode::I4_SYM,
              kernel_config.wg_n,
              kernel_config.wg_k,
              kernel_config.sg_n,
              kernel_config.sg_k,
              kernel_config.k_stride,
              supproted_gzs[idx],
              0>(
              reinterpret_cast<sycl::half*>(dequant_weight_->data_ptr()),
              reinterpret_cast<uint32_t*>(weight_->data_ptr()),
              nullptr,
              reinterpret_cast<sycl::half*>(weight_scl_->data_ptr()),
              n_,
              k_)};
          DPCPP_Q_SUBMIT_CGFS(q, cfgs);
        } else {
          auto cfgs = {xetla_dequantize_int4_weight<
              sycl::half,
              gpu::xetla::quant_mode::I4_ASYM,
              kernel_config.wg_n,
              kernel_config.wg_k,
              kernel_config.sg_n,
              kernel_config.sg_k,
              kernel_config.k_stride,
              supproted_gzs[idx],
              0>(
              reinterpret_cast<sycl::half*>(dequant_weight_->data_ptr()),
              reinterpret_cast<uint32_t*>(weight_->data_ptr()),
              reinterpret_cast<uint32_t*>(weight_zp_ptr_->data_ptr()),
              reinterpret_cast<sycl::half*>(weight_scl_->data_ptr()),
              n_,
              k_)};
          DPCPP_Q_SUBMIT_CGFS(q, cfgs);
        }
      } else {
        dispatch<idx + 1>(q);
      }
    }
  }

  void run() {
    auto& q = torch_ipex::xpu::dpcpp::dpcppGetCurrentQueue();
    dispatch(q);
  }
};
