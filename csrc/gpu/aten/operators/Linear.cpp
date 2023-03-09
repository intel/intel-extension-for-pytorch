#include "Linear.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

using namespace impl;

struct LinearConverter {
  LinearConverter() {
    is_fused_ = false;
  }

  // linear with post-ops
  template <typename Func>
  Tensor& call(
      Tensor& result,
      const Tensor& input,
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      Func func) {
    Attr attr = func();
    Tensor _bias = bias.has_value() ? bias.value() : at::Tensor();
    Tensor _input =
        input.dim() <= 2 ? input : xpu::oneDNN::contiguous_if_needed(input);
    return matmul_fusion_variants(
        result, _input, weight, false, attr, is_fused_, _bias);
  }

  bool is_fused() {
    return is_fused_;
  }

  bool is_fused_;
};

#define IPEX_LINEAR_DEFINATION(func)                                       \
  Tensor linear_##func(                                                    \
      const Tensor& input,                                                 \
      const Tensor& weight,                                                \
      const c10::optional<Tensor>& bias) {                                 \
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
    Tensor result;                                                         \
    return linear_wrapper.call(result, input, weight, bias, post_op);      \
  }

#define IPEX_LINEAR_BINARY_DEFINATION(func)                                \
  Tensor linear_binary_##func(                                             \
      const Tensor& input,                                                 \
      const Tensor& weight,                                                \
      const c10::optional<Tensor>& bias,                                   \
      const Tensor& binary) {                                              \
    RECORD_FUNCTION(                                                       \
        "linear_binary_" #func,                                            \
        std::vector<c10::IValue>({input, weight, bias}));                  \
    auto linear_wrapper = LinearConverter();                               \
    auto post_op = [=]() {                                                 \
      Attr attr;                                                           \
      attr.append_scale_binary(attr.kind_with_binary_##func, binary, 1.f); \
      return attr;                                                         \
    };                                                                     \
    Tensor result;                                                         \
    result = linear_wrapper.call(result, input, weight, bias, post_op);    \
    if (!linear_wrapper.is_fused()) {                                      \
      result = at::func(result, binary);                                   \
    }                                                                      \
    return result;                                                         \
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
IPEX_LINEAR_DEFINATION(mish)

IPEX_LINEAR_BINARY_DEFINATION(mul)
IPEX_LINEAR_BINARY_DEFINATION(div)
IPEX_LINEAR_BINARY_DEFINATION(min)
IPEX_LINEAR_BINARY_DEFINATION(max)
IPEX_LINEAR_BINARY_DEFINATION(eq)
IPEX_LINEAR_BINARY_DEFINATION(ne)
IPEX_LINEAR_BINARY_DEFINATION(ge)
IPEX_LINEAR_BINARY_DEFINATION(gt)
IPEX_LINEAR_BINARY_DEFINATION(le)
IPEX_LINEAR_BINARY_DEFINATION(lt)

Tensor linear_hardswish(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  RECORD_FUNCTION(
      "linear_hardswish", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f / 6.f,
        /* beta */ 1.f / 2.f,
        attr.kind_with_hardswish);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_log_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  RECORD_FUNCTION(
      "linear_log_sigmoid", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ -1.f,
        /* beta */ 0.f,
        attr.kind_with_soft_relu);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_silu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  RECORD_FUNCTION(
      "linear_silu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f,
        /* beta */ 0.f,
        attr.kind_with_swish);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_scalar_mul(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Scalar scalar) {
  RECORD_FUNCTION(
      "linear_scalar_mul", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ scalar.toFloat(),
        /* beta */ 0.f,
        attr.kind_with_linear);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_scalar_div(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Scalar scalar) {
  TORCH_INTERNAL_ASSERT(scalar.toFloat() != 0, "div zero in linear_scalar_div");
  RECORD_FUNCTION(
      "linear_scalar_div", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f / scalar.toFloat(),
        /* beta */ 0.f,
        attr.kind_with_linear);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_scalar_add(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Scalar scalar,
    Scalar scale) {
  RECORD_FUNCTION(
      "linear_scalar_add", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f,
        /* beta */ scalar.toFloat() * scale.toFloat(),
        attr.kind_with_linear);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_scalar_sub(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Scalar scalar,
    Scalar scale) {
  RECORD_FUNCTION(
      "linear_scalar_sub", std::vector<c10::IValue>({input, weight, bias}));
  return linear_scalar_add(input, weight, bias, scalar, -scale);
}

Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
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
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_hardsigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  RECORD_FUNCTION(
      "linear_hardsigmoid", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f / 6.,
        /* beta */ 1.f / 2.,
        attr.kind_with_hardsigmoid);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_pow(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Scalar exponent) {
  RECORD_FUNCTION(
      "linear_pow", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 1.f,
        /* beta */ exponent.toFloat(),
        attr.kind_with_pow);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_leaky_relu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Scalar negative_slope) {
  RECORD_FUNCTION(
      "linear_leaky_relu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ negative_slope.toFloat(),
        /* beta */ 0.f,
        attr.kind_with_relu);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_hardtanh(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Scalar minval,
    Scalar maxval) {
  RECORD_FUNCTION(
      "linear_hardtanh", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ minval.toFloat(),
        /* beta */ maxval.toFloat(),
        attr.kind_with_clip);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

Tensor linear_elu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  RECORD_FUNCTION(
      "linear_elu", std::vector<c10::IValue>({input, weight, bias}));
  auto linear_wrapper = LinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ alpha.toFloat(),
        /* beta */ 1.f,
        attr.kind_with_elu);
    return attr;
  };
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

// accumul = (input * weight + bias + alpha * accumul)
// accumul = Add(Linear(input, weight), accumul)
// pattern: aten::linear, aten::add_ (inplace accumul)
// accumul will be inplaced add
Tensor linear_sum(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Tensor& accumul,
    Scalar alpha) {
  RECORD_FUNCTION(
      "linear_sum", std::vector<c10::IValue>({input, weight, bias}));
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_sum(alpha.to<float>());
    return attr;
  };
  auto linear_wrapper = LinearConverter();
  return linear_wrapper.call(accumul, input, weight, bias, post_op);
}

// result = (input * weight + bias + alpha * accumul)
// result = Add(Linear(input, weight), accumul)
// pattern: aten::linear, aten::add (outplace accumul tensor)
// pattern: aten::linear, aten::add_ (inplace conv output and outplace accumul)
// accumul will be outplaced add
Tensor linear_binary_add(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    Tensor& accumul,
    Scalar alpha) {
  RECORD_FUNCTION(
      "linear_binary_add", std::vector<c10::IValue>({input, weight, bias}));
  auto post_op = [=]() {
    Attr attr;
    attr.append_scale_binary(
        attr.kind_with_binary_add, accumul, alpha.to<float>());
    return attr;
  };
  auto linear_wrapper = LinearConverter();
  Tensor result;
  result = linear_wrapper.call(result, input, weight, bias, post_op);

  if (!linear_wrapper.is_fused()) {
    result = at::AtenIpexTypeXPU::add(result, accumul, alpha.to<float>());
  }
  return result;
}

// outplace the binary tensor
Tensor linear_binary_sub(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    at::Tensor& binary,
    at::Scalar alpha) {
  RECORD_FUNCTION(
      "linear_binary_sub",
      std::vector<c10::IValue>({input, weight, bias, binary}));
  return linear_binary_add(input, weight, bias, binary, -alpha);
}

Tensor dpcpp_linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  auto post_op = [=]() {
    Attr attr;
    return attr;
  };
  auto linear_wrapper = LinearConverter();
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
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
  IPEX_OP_REGISTER_LINEAR(binary_sub);
  IPEX_OP_REGISTER_LINEAR(binary_mul);
  IPEX_OP_REGISTER_LINEAR(binary_div);
  IPEX_OP_REGISTER_LINEAR(binary_min);
  IPEX_OP_REGISTER_LINEAR(binary_max);
  IPEX_OP_REGISTER_LINEAR(binary_eq);
  IPEX_OP_REGISTER_LINEAR(binary_ne);
  IPEX_OP_REGISTER_LINEAR(binary_ge);
  IPEX_OP_REGISTER_LINEAR(binary_gt);
  IPEX_OP_REGISTER_LINEAR(binary_le);
  IPEX_OP_REGISTER_LINEAR(binary_lt);
  IPEX_OP_REGISTER("linear_add", linear_binary_add);
  IPEX_OP_REGISTER("linear_binary_mul.Scalar", linear_scalar_mul);
  IPEX_OP_REGISTER("linear_binary_div.Scalar", linear_scalar_div);
  IPEX_OP_REGISTER("linear_add.Scalar", linear_scalar_add);
  IPEX_OP_REGISTER("linear_sum.Scalar", linear_scalar_add);
  IPEX_OP_REGISTER("linear_binary_sub.Scalar", linear_scalar_sub);
}
} // namespace AtenIpexTypeXPU
} // namespace at
