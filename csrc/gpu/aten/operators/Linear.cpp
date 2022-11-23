#include "Linear.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

using namespace impl;

struct LinearConverter {
  LinearConverter() {
    is_fused_ = false;
  }

  // linear_out with post-ops
  template <typename Func>
  void call(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      Tensor& result,
      Func func) {
    // For those weight, which has been prepacked for linear through
    // torch.xpu.optimize, the shape both in weight tensorimpl and metadata are
    // matched, so there is no permution operation happened for context, thus
    // the permution().size() will be zero, which means the transpose has
    // happened through prepack, so there is no need to do transpose again here.
    // If the permutation has elements, that means the shape in weight wrapper
    // is not matched with the meta context, so the tranpose is not happened and
    // it is needed here.
    auto weight_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
    bool m2_trans = (!weight_ctx.is_plain()) &&
        (weight_ctx.permution().size() ? false : true);

    Attr attr = func();
    Tensor _bias = bias.defined() ? bias : at::Tensor();
    if (input.dim() == 2) {
      is_fused_ = true;
      impl::onednn_matmul(result, input, weight, _bias, result, m2_trans, attr);
      return;
    }

    if (input.dim() == 3 && input.is_contiguous()) {
      // Also hit the fused path for contiguous 3D input.
      is_fused_ = true;
      const auto input_sizes = input.sizes();
      const auto weight_sizes = weight.sizes();
      std::vector<int64_t> output_sizes = {
          input_sizes[0], input_sizes[1], weight_sizes[1]};
      auto input_view =
          input.view({input_sizes[0] * input_sizes[1], input_sizes[2]});
      if (result.defined()) {
        result =
            result.view({output_sizes[0] * output_sizes[1], output_sizes[2]});
      }
      impl::onednn_matmul(
          result, input_view, weight, _bias, result, m2_trans, attr);
      result = result.view({input_sizes[0], input_sizes[1], result.size(1)});
      return;
    }

    result = at::matmul(input, weight.t());
    if (bias.defined()) {
      result.add_(bias);
    }
  }

  bool is_fused() {
    return is_fused_;
  }

  bool is_fused_;
};

#define IPEX_LINEAR_DEFINATION(func)                                       \
  Tensor linear_##func(                                                    \
      const Tensor& input, const Tensor& weight, const Tensor& bias) {     \
    RECORD_FUNCTION(                                                       \
        "linear_" #func, std::vector<c10::IValue>({input, weight, bias})); \
    auto linear_wrapper = LinearConverter();                               \
    auto post_op = [=]() {                                                 \
      Attr attr;                                                           \
      attr.append_post_eltwise(                                            \
          /* scale */ 1.f,                                                 \
          /* alpha */ 0.f,                                                 \
          /* beta */ 0.f,                                                  \
          attr.kind_with_##func);                                          \
      return attr;                                                         \
    };                                                                     \
    Tensor output;                                                         \
    linear_wrapper.call(input, weight, bias, output, post_op);             \
    if (!linear_wrapper.is_fused()) {                                      \
      output = at::func(output);                                           \
    }                                                                      \
    return output;                                                         \
  }

IPEX_LINEAR_DEFINATION(sqrt)
IPEX_LINEAR_DEFINATION(abs)
IPEX_LINEAR_DEFINATION(tanh)
IPEX_LINEAR_DEFINATION(square)
IPEX_LINEAR_DEFINATION(exp)
IPEX_LINEAR_DEFINATION(log)
IPEX_LINEAR_DEFINATION(round)
IPEX_LINEAR_DEFINATION(sigmoid)
IPEX_LINEAR_DEFINATION(relu)
IPEX_LINEAR_DEFINATION(hardswish)
IPEX_LINEAR_DEFINATION(mish)

Tensor linear_silu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  RECORD_FUNCTION(
      "linear_silu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ 1.f,
        /* beta */ 0.f,
        attr.kind_with_swish);
    return attr;
  };
  Tensor output;
  linear_wrapper.call(input, weight, bias, output, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::silu(output);
  }
  return output;
}

Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    c10::string_view approximate) {
  RECORD_FUNCTION(
      "linear_gelu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    algorithm algo;
    if (approximate == "none") {
      algo = attr.kind_with_gelu_erf;
    } else if (approximate == "tanh") {
      algo = attr.kind_with_gelu_tanh;
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
    }
    attr.append_post_eltwise(1.0f, 0.0f, 0.0f, algo);
    return attr;
  };
  Tensor output;
  linear_wrapper.call(input, weight, bias, output, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::AtenIpexTypeXPU::gelu_out(output, approximate, output);
  }
  return output;
}

Tensor linear_log_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  RECORD_FUNCTION(
      "linear_logsigmoid", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ 1.f,
        /* beta */ 0.f,
        attr.kind_with_logsigmoid);
    return attr;
  };
  Tensor output;
  linear_wrapper.call(input, weight, bias, output, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::log_sigmoid(output);
  }
  return output;
}

Tensor linear_hardsigmoid(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  RECORD_FUNCTION(
      "linear_hardsigmoid", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ 1.f / 6.,
        /* beta */ 1.f / 2.,
        attr.kind_with_hardsigmoid);
    return attr;
  };
  Tensor output;
  linear_wrapper.call(input, weight, bias, output, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::hardsigmoid(output);
  }
  return output;
}

Tensor linear_pow(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar exponent) {
  RECORD_FUNCTION(
      "linear_pow", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ 1.f,
        /* beta */ exponent.toFloat(),
        attr.kind_with_pow);
    return attr;
  };
  Tensor output;
  linear_wrapper.call(input, weight, bias, output, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::pow(output, exponent);
  }
  return output;
}

Tensor linear_leaky_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar negative_slope) {
  RECORD_FUNCTION(
      "linear_leaky_relu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ negative_slope.toFloat(),
        /* beta */ 0.f,
        attr.kind_with_relu);
    return attr;
  };
  Tensor output;
  linear_wrapper.call(input, weight, bias, output, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::leaky_relu(output, negative_slope);
  }
  return output;
}

Tensor linear_hardtanh(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar minval,
    Scalar maxval) {
  RECORD_FUNCTION(
      "linear_hardtanh", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ minval.toFloat(),
        /* beta */ maxval.toFloat(),
        attr.kind_with_clip);
    return attr;
  };
  Tensor output;
  linear_wrapper.call(input, weight, bias, output, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::hardtanh(output);
  }
  return output;
}

Tensor linear_elu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  RECORD_FUNCTION(
      "linear_elu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ alpha.toFloat(),
        /* beta */ 1.f,
        attr.kind_with_elu);
    return attr;
  };
  Tensor output;
  linear_wrapper.call(input, weight, bias, output, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::elu(output, alpha, scale, input_scale);
  }
  return output;
}

// result = (input * weight + bias + alpha * accumul)
Tensor linear_sum(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& accumul,
    Scalar alpha) {
  RECORD_FUNCTION(
      "linear_sum", std::vector<c10::IValue>({input, weight, bias}));
  const auto input_sizes = input.sizes();
  const auto weight_sizes = weight.sizes();
  std::vector<int64_t> output_sizes = {
      input_sizes[0], input_sizes[1], weight_sizes[1]};

  Tensor output = at::empty(output_sizes, input.options());
  bool can_be_fused;
  Attr attr = get_onednn_linear_sum_attr(
      input, weight, accumul, output, alpha.to<float>(), can_be_fused);
  auto post_op = [=]() { return attr; };
  auto linear_wrapper = LinearConverter();
  linear_wrapper.call(input, weight, bias, output, post_op);

  bool is_success_fused = can_be_fused && linear_wrapper.is_fused();
  if (!is_success_fused) {
    output = at::AtenIpexTypeXPU::add(output, accumul, alpha.to<float>());
  }
  return output;
}

Tensor& dpcpp_linear_out(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output) {
  auto post_op = [=]() {
    Attr attr;
    return attr;
  };
  auto linear_wrapper = LinearConverter();
  linear_wrapper.call(input, weight, bias, output, post_op);
  return output;
}

// IPEX customer linear for weight prepack
Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  RECORD_FUNCTION("linear", std::vector<c10::IValue>({input, weight}));
  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(c10::in_place);
  auto weight_ctx =
      at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
  auto is_weight_plain = weight_ctx.is_plain();

  // For those weight, which has been prepacked for linear through
  // torch.xpu.optimize, the shape both in weight tensorimpl and metadata are
  // matched, so there is no permution operation happened for context, thus the
  // permution().size() will be zero, which means the transpose has happened
  // through prepack, so there is no need to do transpose again here.
  // If the permutation has elements, that means the shape in weight wrapper is
  // not matched with the meta context, so the tranpose is not happened and it
  // is needed here.
  auto is_transposed = weight_ctx.permution().size() ? false : true;

  if (input.dim() == 2 && bias->defined()) {
    // if weight is block format and tranposed, the transpose here is merged
    // into linear weight prepack, so no need to do transpose.
    return at::addmm(
        *bias,
        input,
        ((!is_weight_plain) && is_transposed) ? weight : weight.t());
  }

  if (input.dim() == 3 && bias->defined() && input.is_contiguous()) {
    // Also hit the fused path for contiguous 3D input.
    const auto input_sizes = input.sizes();
    const auto result = at::addmm(
        *bias,
        input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}),
        ((!is_weight_plain) && is_transposed) ? weight : weight.t());
    return result.view({input_sizes[0], input_sizes[1], result.size(1)});
  }

  auto output = at::matmul(
      input, ((!is_weight_plain) && is_transposed) ? weight : weight.t());
  if (bias->defined()) {
    output.add_(*bias);
  }
  return output;
}

Tensor& linear_out(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    Tensor& output) {
  RECORD_FUNCTION("linear_out", std::vector<c10::IValue>({input, weight}));
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  auto weight_ctx =
      at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
  auto is_weight_plain = weight_ctx.is_plain();

  // For those weight, which has been prepacked for linear through
  // torch.xpu.optimize, the shape both in weight tensorimpl and metadata are
  // matched, so there is no permution operation happened for context, thus the
  // permution().size() will be zero, which means the transpose has happened
  // through prepack, so there is no need to do transpose again here.
  // If the permutation has elements, that means the shape in weight wrapper is
  // not matched with the meta context, so the tranpose is not happened and it
  // is needed here.
  auto is_transposed = weight_ctx.permution().size() ? false : true;

  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm_out(
        output,
        *bias,
        input,
        ((!is_weight_plain) && is_transposed) ? weight : weight.t());
  }
  output = at::matmul_out(
      output,
      input,
      ((!is_weight_plain) && is_transposed) ? weight : weight.t());
  if (bias->defined()) {
    output.add_(*bias);
  }
  return output;
}

// Here register linear and linear_out on both XPU and AutogradXPU. Firstly,
// with torch inference mode, the all autograd-kind dispatch key will be
// excluded so it will go into aten native linear, so we need to register XPU.
// Secondly, linear is a compound op and torch will not build autograd graph for
// it. If we only register XPU backend, the torch will stop this behavior unless
// you register an autograd one.
TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("linear", TORCH_FN(linear));
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("linear_out", TORCH_FN(linear_out));
}

TORCH_LIBRARY_IMPL(aten, AutogradXPU, m) {
  m.impl("linear", TORCH_FN(linear));
}

TORCH_LIBRARY_IMPL(aten, AutogradXPU, m) {
  m.impl("linear_out", TORCH_FN(linear_out));
}

#define IPEX_OP_REGISTER_LINEAR(op) \
  IPEX_OP_REGISTER("linear_" #op, linear_##op);

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_LINEAR(sigmoid);
  IPEX_OP_REGISTER_LINEAR(relu);
  IPEX_OP_REGISTER_LINEAR(sqrt);
  IPEX_OP_REGISTER_LINEAR(abs);
  IPEX_OP_REGISTER_LINEAR(tanh);
  IPEX_OP_REGISTER_LINEAR(square);
  IPEX_OP_REGISTER_LINEAR(exp);
  IPEX_OP_REGISTER_LINEAR(log);
  IPEX_OP_REGISTER_LINEAR(round);
  IPEX_OP_REGISTER_LINEAR(log_sigmoid);
  IPEX_OP_REGISTER_LINEAR(hardswish);
  IPEX_OP_REGISTER_LINEAR(mish);
  IPEX_OP_REGISTER_LINEAR(silu);
  IPEX_OP_REGISTER_LINEAR(hardsigmoid);
  IPEX_OP_REGISTER_LINEAR(leaky_relu);
  IPEX_OP_REGISTER_LINEAR(pow);
  IPEX_OP_REGISTER_LINEAR(hardtanh);
  IPEX_OP_REGISTER_LINEAR(elu);
  IPEX_OP_REGISTER_LINEAR(sum);
  IPEX_OP_REGISTER_LINEAR(gelu);
}
} // namespace AtenIpexTypeXPU
} // namespace at
