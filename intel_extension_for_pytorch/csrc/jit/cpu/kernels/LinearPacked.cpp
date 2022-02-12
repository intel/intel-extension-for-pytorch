#include "LinearPacked.h"
#include "csrc/aten/cpu/Linear.h"
#include "csrc/aten/cpu/WeightPack.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/utils/ipex_op_profile.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace linear {

c10::intrusive_ptr<LinearOpContext> createLinearPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    int64_t out_features,
    int64_t in_features,
    int64_t batch_size,
    bool weight_is_packed) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::createLinearPrePackOpContext",
      std::vector<c10::IValue>({}));

  return IpexLinearOpContext::create_context(
      std::move(weight),
      std::move(bias),
      out_features,
      in_features,
      batch_size,
      weight_is_packed);
}

at::Tensor linear_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::linear_run", std::vector<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t());
}

at::Tensor linear_relu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::linear_relu_run", std::vector<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t::fuse_relu());
}

at::Tensor linear_gelu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::linear_gelu_run", std::vector<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t::fuse_gelu());
}

at::Tensor linear_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::linear_add_run", std::vector<c10::IValue>({}));

  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(input, accumu, ideep::attr_t::fuse_sum(scale));
}

ContextLinear create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const int64_t out_features,
    const int64_t in_features,
    const int64_t batch_size,
    const bool weight_is_packed) {
  auto weight_dtype = get_mkldnn_dtype(weight.scalar_type());
  ideep::tensor parcked_weight;
  auto packed_desc = ideep::inner_product_forward::expected_weights_desc(
      {out_features, in_features},
      {batch_size, in_features},
      /* weight dtype */ weight_dtype,
      /* src dtype */ weight_dtype);
  parcked_weight.init(packed_desc);
  if (!weight_is_packed) {
    auto weight_ = weight.contiguous();
    ideep::tensor w = itensor_view_from_dense(weight_);
    parcked_weight.feed_from(w);
  } else {
    auto w = get_linear_packed_weight(weight, out_features, in_features);
    parcked_weight.feed_from(w);
  }
  return ContextLinear{
      std::move(parcked_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
  };
}

at::Tensor run(
    const ContextLinear& context,
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  auto input_ = input.contiguous();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(context.bias_);
  const at::Tensor& bias = *bias_maybe_owned;
  return linear_kernel(input_, context.weight_packed_, bias, attr);
}

at::Tensor& run(
    const ContextLinear& context,
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr) {
  auto input_ = input.contiguous();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(context.bias_);
  const at::Tensor& bias = *bias_maybe_owned;
  linear_kernel_output(input_, context.weight_packed_, bias, accumu, attr);
  return accumu;
}

} // namespace linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex