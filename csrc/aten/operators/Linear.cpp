#include "BlasImpl.h"

namespace at {
namespace AtenIpexTypeXPU {

using namespace impl;

struct LinearConverter {
  LinearConverter() {
    is_fused_ = false;
  }

  // linear with accumul and post-ops
  template <typename Func>
  Tensor call(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      const Tensor& accumul,
      Func func) {
    const auto input_sizes = input.sizes();
    const auto weight_sizes = weight.sizes();
    std::vector<int64_t> output_sizes = {
        input_sizes[0], input_sizes[1], weight_sizes[1]};
    auto result = at::empty({0}, input.options());

    Attr attr = func();
    Tensor _bias = bias.defined() ? bias : at::Tensor();
    if (input.dim() == 2) {
      is_fused_ = true;
      impl::onednn_matmul(result, input, weight, _bias, accumul, false, attr);
      return result;
    }

    if (input.dim() == 3 && input.is_contiguous()) {
      // Also hit the fused path for contiguous 3D input.
      is_fused_ = true;
      const auto input_sizes = input.sizes();
      auto input_view =
          input.view({input_sizes[0] * input_sizes[1], input_sizes[2]});
      auto accumul_view = accumul;
      if (accumul.defined()) {
        accumul_view =
            accumul.view({output_sizes[0] * output_sizes[1], output_sizes[2]});
      }
      impl::onednn_matmul(
          result, input_view, weight, _bias, accumul_view, false, attr);
      return result.view({input_sizes[0], input_sizes[1], result.size(1)});
    }

    auto output = at::matmul(input, weight.t());
    if (bias.defined()) {
      output.add_(bias);
    }
    return output;
  }

  // linear with post-ops
  template <typename Func>
  Tensor call(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      Func func) {
    Tensor accumul = at::Tensor();
    return call(input, weight, bias, accumul, func);
  }

  // linear
  Tensor call(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    auto post_op = [=]() {
      Attr attr;
      return attr;
    };
    return call(input, weight, bias, post_op);
  }

  bool is_fused() {
    return is_fused_;
  }

  bool is_fused_;
};

Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  RECORD_FUNCTION(
      "linear_gelu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_gelu);
    return attr;
  };
  Tensor output = linear_wrapper.call(input, weight, bias, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::AtenIpexTypeXPU::gelu(output);
  }
  return output;
}

Tensor linear_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  RECORD_FUNCTION(
      "linear_relu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_relu);
    return attr;
  };
  Tensor output = linear_wrapper.call(input, weight, bias, post_op);
  if (!linear_wrapper.is_fused()) {
    output = at::AtenIpexTypeXPU::relu(output);
  }
  return output;
}

Tensor linear_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  RECORD_FUNCTION(
      "linear_sigmoid", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* gelu_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_sigmoid);
    return attr;
  };
  Tensor output = linear_wrapper.call(input, weight, bias, post_op);
  if (!linear_wrapper.is_fused()) {
    at::AtenIpexTypeXPU::sigmoid_out(output, output);
  }
  return output;
}

// result = (input * weight + bias + alpha * accumul)
Tensor linear_add(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& accumul,
    Scalar alpha) {
  RECORD_FUNCTION(
      "linear_add", std::vector<c10::IValue>({input, weight, bias}));
  const auto input_sizes = input.sizes();
  const auto weight_sizes = weight.sizes();
  std::vector<int64_t> output_sizes = {
      input_sizes[0], input_sizes[1], weight_sizes[1]};

  Tensor result;
  auto linear_wrapper = LinearConverter();
  bool can_be_fused = (accumul.sizes().vec() == output_sizes);
  if (can_be_fused) {
    auto post_op = [=]() {
      Attr attr;
      attr.append_post_sum(/* sum_scale */ alpha.to<float>());
      return attr;
    };
    result = linear_wrapper.call(input, weight, bias, accumul, post_op);
  } else {
    result = linear_wrapper.call(input, weight, bias);
  }

  bool is_success_fused = can_be_fused && linear_wrapper.is_fused();
  if (!is_success_fused) {
    result = at::AtenIpexTypeXPU::add(result, accumul, alpha.to<float>());
  }
  return result;
}

// IPEX customer linear for weight prepack
Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
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

} // namespace AtenIpexTypeXPU
} // namespace at
