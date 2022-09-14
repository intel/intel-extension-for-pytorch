#include "LinearPacked.h"
#include <ideep.hpp>
#include "aten/Linear.h"
#include "aten/WeightPack.h"
#include "ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace linear {

#define DEFINE_LINEAR_UNARY_ELTWISE_RUN(FUSED_OP)              \
  at::Tensor linear_##FUSED_OP##_run(                          \
      const at::Tensor& input,                                 \
      const c10::intrusive_ptr<LinearOpContext>& op_context) { \
    RECORD_FUNCTION(                                           \
        "ipex_prepack::linear_" #FUSED_OP "_run",              \
        c10::ArrayRef<c10::IValue>({}));                       \
    return op_context->run(                                    \
        input,                                                 \
        ideep::attr_t::fuse_##FUSED_OP().set_fpmath_mode(      \
            torch_ipex::fpmath_mode));                         \
  }

c10::intrusive_ptr<LinearOpContext> createLinearPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size) {
  RECORD_FUNCTION(
      "ipex_prepack::createLinearPrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));

  return IpexLinearOpContext::create_context(
      std::move(weight), std::move(bias), batch_size);
}

at::Tensor linear_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  RECORD_FUNCTION("ipex_prepack::linear_run", c10::ArrayRef<c10::IValue>({}));

  return op_context->run(input, ideep::attr_t(torch_ipex::fpmath_mode));
}

DEFINE_LINEAR_UNARY_ELTWISE_RUN(relu);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(sigmoid);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(swish);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(tanh);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(mish);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(abs);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(exp);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(hardswish);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(square);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(log);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(round);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(sqrt);
DEFINE_LINEAR_UNARY_ELTWISE_RUN(hardsigmoid);

at::Tensor linear_leaky_relu_run(
    const at::Tensor& input,
    at::Scalar alpha,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::linear_leaky_relu_run", c10::ArrayRef<c10::IValue>({}));
  auto alpha_value = alpha.to<float>();
  return op_context->run(
      input,
      ideep::attr_t::fuse_relu(1.0, alpha_value)
          .set_fpmath_mode(torch_ipex::fpmath_mode));
}

at::Tensor linear_hardtanh_run(
    const at::Tensor& input,
    at::Scalar lower_bound,
    at::Scalar upper_bound,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::linear_hardtanh_run", c10::ArrayRef<c10::IValue>({}));
  auto lower_bound_value = lower_bound.to<float>();
  auto upper_bound_value = upper_bound.to<float>();
  return op_context->run(
      input,
      ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value)
          .set_fpmath_mode(torch_ipex::fpmath_mode));
}

at::Tensor linear_elu_run(
    const at::Tensor& input,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::linear_elu_run", c10::ArrayRef<c10::IValue>({}));
  auto alpha_value = alpha.to<float>();
  auto scale_value = scale.to<float>();
  auto input_scale_value = input_scale.to<float>();
  return op_context->run(
      input,
      ideep::attr_t::fuse_elu(scale_value, alpha_value, input_scale_value)
          .set_fpmath_mode(torch_ipex::fpmath_mode));
}

at::Tensor linear_pow_run(
    const at::Tensor& input,
    at::Scalar exponent,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::linear_pow_run", c10::ArrayRef<c10::IValue>({}));
  auto exponent_value = exponent.to<float>();
  return op_context->run(
      input,
      ideep::attr_t::fuse_pow(1.0, 1.0, exponent_value)
          .set_fpmath_mode(torch_ipex::fpmath_mode));
}

at::Tensor linear_gelu_run(
    const at::Tensor& input,
    const c10::string_view approximate,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::linear_gelu_run", c10::ArrayRef<c10::IValue>({}));
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
      input,
      ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type)
          .set_fpmath_mode(torch_ipex::fpmath_mode));
}

at::Tensor linear_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::linear_add_run", c10::ArrayRef<c10::IValue>({}));

  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(
      input,
      accumu,
      ideep::attr_t::fuse_sum(scale).set_fpmath_mode(torch_ipex::fpmath_mode));
}

at::Tensor linear_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::linear_add_relu_run", c10::ArrayRef<c10::IValue>({}));

  auto scale = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  return op_context->run(
      input,
      accumu,
      ideep::attr_t::residual(scale).set_fpmath_mode(torch_ipex::fpmath_mode));
}

ContextLinear create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size) {
  auto out_features = weight.size(0);
  auto in_features = weight.size(1);
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
  TORCH_CHECK(
      input.size(input.dim() - 1) == context.weight_packed_.get_dims()[1],
      "Check the shapes of mat1 and mat2, they cannot be multiplied!");
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
  TORCH_CHECK(
      input.size(input.dim() - 1) == context.weight_packed_.get_dims()[1],
      "Check the shapes of mat1 and mat2, they cannot be multiplied!");
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

} // namespace linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
