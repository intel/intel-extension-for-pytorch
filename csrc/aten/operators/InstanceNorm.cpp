#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MatrixRef.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/record_function.h>
#include <dnnl.hpp>
#include <oneDNN/oneDNN.h>
#include <torch/autograd.h>
#include <torch/custom_class.h>
#include "comm/ATDispatch.h"

#include "comm/ParamUtils.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;
using namespace torch::autograd;

namespace at {
namespace AtenIpexTypeXPU {

static inline Tensor repeat_if_defined(
    c10::optional<Tensor> const& t,
    int64_t repeat) {
  if (t.has_value()) {
    return t.value().repeat(repeat);
  }
  // If there is not value for t, return a empty Tensor
  return Tensor();
}

Tensor instance_norm(
    at::Tensor const& input,
    c10::optional<Tensor> const& weight,
    c10::optional<Tensor> const& bias,
    c10::optional<Tensor> const& running_mean,
    c10::optional<Tensor> const& running_var,
    bool use_input_stats,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  TORCH_CHECK(
      use_input_stats || (running_mean.has_value() && running_var.has_value()),
      "Expected running_mean and running_var to be defined when use_input_stats is false");

  std::vector<int64_t> shape = input.sizes().vec();
  int64_t b = input.size(0);
  int64_t c = input.size(1);
  shape[1] = b * c;
  shape[0] = 1;

  Tensor weight_ = repeat_if_defined(weight, b);
  Tensor bias_ = repeat_if_defined(bias, b);
  Tensor running_mean_ = repeat_if_defined(running_mean, b);
  Tensor running_var_ = repeat_if_defined(running_var, b);

  auto input_reshaped = input.reshape(shape);
  input_reshaped = is_smf_channels_last(input)
      ? input_reshaped.to(get_cl_tag_by_ndim(shape.size()))
      : input_reshaped.to(at::MemoryFormat::Contiguous);

  auto out = at::batch_norm(
      input_reshaped,
      weight_,
      bias_,
      running_mean_,
      running_var_,
      use_input_stats,
      momentum,
      eps,
      cudnn_enabled);

  // we alias running_mean and running_var because they are const but we want to
  // modify their data
  if (running_mean.has_value()) {
    at::alias(running_mean.value())
        .copy_(running_mean_.view({b, c}).mean(0, false));
  }
  if (running_var.has_value()) {
    at::alias(running_var.value())
        .copy_(running_var_.view({b, c}).mean(0, false));
  }

  return out.reshape(input.sizes()).to(input.suggest_memory_format());
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("instance_norm", TORCH_FN(AtenIpexTypeXPU::instance_norm));
}
} // namespace AtenIpexTypeXPU
} // namespace at
