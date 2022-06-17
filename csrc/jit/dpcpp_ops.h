#pragma once
#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <oneDNN/oneDNN.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace torch {
namespace jit {
namespace xpu {

namespace {

template <typename T, int N>
struct TypeSelector {
  template <typename... Args>
  void extract_type(Args... args) {
    return;
  }

  template <typename... Args>
  void extract_type(T& type, Args... args) {
    container_.push_back(type);
    extract_type(args...);
  }

  template <typename U, typename... Args>
  void extract_type(U type, Args... args) {
    extract_type(args...);
  }

  at::ArrayRef<T> retrive_types() {
    return at::ArrayRef<T>(container_.begin(), container_.end());
  }

  at::SmallVector<T, N> container_;
};

} // namespace

template <typename Func>
struct JitFusionProxy {
  template <typename... Args>
  at::Tensor operator()(Func func, std::string str, Args... args) {
    RECORD_FUNCTION(str, std::vector<c10::IValue>({args...}));
    TypeSelector<at::Tensor, sizeof...(args)> selector;
    selector.extract_type(args...);
    auto iter = std::find(to_plain_list_.begin(), to_plain_list_.end(), str);
    if (iter != to_plain_list_.end()) {
      std::for_each(
          selector.retrive_types().begin(),
          selector.retrive_types().end(),
          AtenIpexTypeXPU::to_plain_if_needed_);
    }
    const OptionalDeviceGuard device_guard(device_of(selector.retrive_types()));
    return func(args...);
  }
  const std::vector<std::string> to_plain_list_ = {
      "xpu::softplus_tanh",
      "xpu::softplus_tanh_mul"};
};

at::Tensor dequant_pixelshuffle(const at::Tensor& self, int64_t upscale_factor);

at::Tensor dequant_pixelshuffle_quant(
    const at::Tensor& self,
    int64_t upscale_factor,
    double scale,
    int64_t zero_pad,
    at::ScalarType dtype);

at::Tensor batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps,
    bool use_dnn);

at::Tensor fold_weight(
    const at::Tensor& weight,
    const at::Tensor& bn_weight,
    const at::Tensor& running_var,
    float eps);

at::Tensor fold_bias(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    float eps);

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from,
    dnnl::memory::format_tag to,
    int64_t groups);

} // namespace xpu
} // namespace jit
} // namespace torch
