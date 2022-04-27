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
    c10::optional<int64_t> batch_size) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::createLinearPrePackOpContext",
      std::vector<c10::IValue>({}));

  return IpexLinearOpContext::create_context(
      std::move(weight),
      std::move(bias),
      out_features,
      in_features,
      batch_size);
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
    const c10::intrusive_ptr<LinearOpContext>& op_context,
    const c10::string_view approximate) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::linear_gelu_run", std::vector<c10::IValue>({}));
  // https://github.com/pytorch/pytorch/pull/61439
  // at::gelu can support tanh approximate now and OneDNN also support it
  // by changing algorithm If there is other type of approximate are added to
  // pytorch while  OneDNN not support it, we might need a fallback path here.
  dnnl::algorithm gelu_type;
  if (approximate == "none") {
    gelu_type = dnnl::algorithm::eltwise_gelu_erf;
  } else if (approximate == "tanh") {
    gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
  } else {
    TORCH_CHECK(
        false, "ipex::linear_gelu_run only support tanh approximate now");
  }
  return op_context->run(
      input, ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type));
}

at::Tensor linear_tanh_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::linear_tanh_run", std::vector<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t::fuse_tanh());
}

at::Tensor linear_sigmoid_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::linear_sigmoid_run", std::vector<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t::fuse_sigmoid());
}

at::Tensor linear_swish_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  IPEX_RECORD_FUNCTION(
      "ipex_prepack::linear_swish_run", std::vector<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t::fuse_swish());
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
    const c10::optional<int64_t> batch_size) {
  ideep::tensor packed_weight;
  auto w = itensor_view_from_dense(weight);
  ideep::dims input_size;
  auto dtype = w.get_data_type();
  ideep::tensor::desc ori_desc(w.get_desc());
  if (batch_size.has_value()) {
    input_size = {batch_size.value(), in_features};
  }
  auto packed_desc = ideep::inner_product_forward::expected_weights_desc(
      {out_features, in_features},
      input_size,
      /* weight dtype */ dtype,
      /* src dtype */ dtype);
  auto at_weight = empty_aten_tensor_from_desc(packed_desc, weight.options());
  if (ideep::data_type::f32 == dtype) {
    packed_weight.init(packed_desc, at_weight.template data_ptr<float>());
  } else {
    packed_weight.init(
        packed_desc, at_weight.template data_ptr<c10::BFloat16>());
  }
  packed_weight.feed_from(w);
  return ContextLinear{
      std::move(ori_desc),
      std::move(packed_weight),
      std::move(at_weight),
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
    ContextLinear& context,
    const at::Tensor& input,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask) {
  return linear_backward_kernel(
      input,
      grad_output,
      context.at_weight_,
      output_mask,
      context.weight_packed_,
      context.bias_);
}

at::Tensor get_at_packed_weight(ContextLinear& context) {
  return context.at_weight_;
}

void set_bias(ContextLinear& context, at::Tensor& bias) {
  context.bias_ = c10::make_optional<at::Tensor>(std::move(bias));
}

void set_weight(ContextLinear& context, at::Tensor& weight) {
  context.at_weight_.copy_(weight);
}

at::Tensor pack(ContextLinear& context, const at::Tensor& tensor) {
  auto ideep_tensor = itensor_view_from_dense(tensor);
  auto dtype = ideep_tensor.get_data_type();
  auto expected_desc = context.weight_packed_.get_desc().to_type(dtype);
  auto packed_at_tensor =
      empty_aten_tensor_from_desc(expected_desc, tensor.options());
  ideep::tensor packed_tensor;
  if (ideep::data_type::f32 == dtype) {
    packed_tensor.init(
        expected_desc, packed_at_tensor.template data_ptr<float>());
  } else {
    packed_tensor.init(
        expected_desc, packed_at_tensor.template data_ptr<c10::BFloat16>());
  }
  packed_tensor.feed_from(ideep_tensor);
  return packed_at_tensor;
}

at::Tensor unpack(ContextLinear& context, const at::Tensor& tensor) {
  auto dtype = get_mkldnn_dtype(tensor.scalar_type());
  auto expected_desc = context.weight_packed_.get_desc().to_type(dtype);
  ideep::tensor blocked_tensor;
  if (ideep::data_type::f32 == dtype) {
    blocked_tensor.init(expected_desc, tensor.template data_ptr<float>());
  } else {
    blocked_tensor.init(
        expected_desc, tensor.template data_ptr<c10::BFloat16>());
  }

  at::Tensor result = at::empty(expected_desc.get_dims(), tensor.options());
  ideep::tensor pub_tensor;
  auto pub_tensor_desc = context.original_desc_.to_type(dtype);
  if (ideep::data_type::f32 == dtype) {
    pub_tensor.init(pub_tensor_desc, result.template data_ptr<float>());
  } else {
    pub_tensor.init(pub_tensor_desc, result.template data_ptr<c10::BFloat16>());
  }
  pub_tensor.feed_from(blocked_tensor);
  return result;
}

void repack_for(ContextLinear& context, int64_t batch_size) {
  auto dtype = context.original_desc_.get_data_type();
  ideep::tensor packed_weight;
  auto packed_desc = ideep::inner_product_forward::expected_weights_desc(
      context.weight_packed_.get_dims(),
      {batch_size, context.weight_packed_.get_dim(1)},
      /* weight dtype */ dtype,
      /* src dtype */ dtype);
  auto new_at_weight =
      empty_aten_tensor_from_desc(packed_desc, context.at_weight_.options());
  if (ideep::data_type::f32 == dtype) {
    packed_weight.init(packed_desc, new_at_weight.template data_ptr<float>());
  } else {
    packed_weight.init(
        packed_desc, new_at_weight.template data_ptr<c10::BFloat16>());
  }
  packed_weight.feed_from(context.weight_packed_);
  context.at_weight_ = new_at_weight;
  context.weight_packed_ = packed_weight;
}

} // namespace linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
