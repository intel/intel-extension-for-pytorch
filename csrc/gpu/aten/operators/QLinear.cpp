#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>

#include <oneDNN/oneDNN.h>
#include <quantized/QUtils.h>
#include <runtime/Utils.h>
#include "InnerProduct.h"
#include "Linear.h"
#include "comm/ParamUtils.h"
#include "utils/CustomOperatorRegistration.h"

using namespace dnnl;
using namespace at::native;
using namespace torch_ipex::xpu::dpcpp;
using namespace torch_ipex::xpu::oneDNN;
using namespace at::AtenIpexTypeXPU::impl;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

struct QLinearConverter {
  QLinearConverter() {
    is_fused_ = false;
  }

  // q_linear with post_ops
  template <typename Func>
  void call(
      Tensor& result,
      Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point,
      at::ScalarType dtype,
      Func func) {
    Attr attr = func();
    auto pack_ptr =
        dynamic_cast<PackedLinearWeightQDPCPP*>(packed_weight.get());
    at::Tensor weight = pack_ptr->weight;
    at::Tensor bias;
    if (pack_ptr->bias_.has_value()) {
      bias = pack_ptr->bias_.value();
    } else {
      bias = Tensor();
    }

    const auto dim_tensor1 = input.dim();
    auto dim_tensor2 = 2;

    // Blocked format path
    auto input_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(input);
    if (!input_ctx.is_plain()) {
      auto sizes = input.sizes();
      auto result_shape = DimVector(sizes.begin(), sizes.end() - 1);
      result_shape.push_back(weight.size(0));
      if (!result.defined()) {
        result = at::_empty_affine_quantized(
            result_shape,
            device(kXPU).dtype(dtype),
            output_scale,
            output_zero_point,
            MemoryFormat::Contiguous);
      }

      is_fused_ = get_onednn_matmul_binary_attr(
          result, attr, dim_tensor1, dim_tensor2, result_shape);
      torch_ipex::xpu::oneDNN::quantized_matmul(
          result, input, weight, bias, false, attr);
      if (result.defined()) {
        set_quantizer_(
            result,
            dpcpp_make_per_tensor_affine_quantizer(
                output_scale, output_zero_point, dtype));
      }
      return;
    }

    // Plain format path
    if (input.dim() == 2) {
      auto sizes = input.sizes();
      auto result_shape = DimVector(sizes.begin(), sizes.end() - 1);
      result_shape.push_back(weight.size(0));
      if (!result.defined()) {
        result = at::_empty_affine_quantized(
            result_shape,
            device(kXPU).dtype(input.scalar_type()),
            output_scale,
            output_zero_point,
            MemoryFormat::Contiguous);
      }
      is_fused_ = get_onednn_matmul_binary_attr(
          result, attr, dim_tensor1, dim_tensor2, result_shape);
      torch_ipex::xpu::oneDNN::quantized_matmul(
          result, input, weight, bias, false, attr);
      return;
    }

    // Plain format path
    if (input.dim() >= 3) {
      auto sizes = input.sizes();
      // output_shape is PyTorch Linear semantic, and can be >= 2-dims
      auto output_shape = DimVector(sizes.begin(), sizes.end() - 1);
      const auto folded_dim = c10::multiply_integers(output_shape);
      output_shape.push_back(weight.size(0));
      // result_shape is for oneDNN matmul, and always in 2-dims
      DimVector result_shape({folded_dim, weight.size(0)});

      auto input_view =
          input.contiguous().view({folded_dim, sizes.back()}).contiguous();
      if (!result.defined()) {
        result = at::_empty_affine_quantized(
            result_shape,
            device(kXPU).dtype(input.scalar_type()),
            output_scale,
            output_zero_point,
            MemoryFormat::Contiguous);
      } else {
        result = result.view(result_shape);
      }
      is_fused_ = get_onednn_matmul_binary_attr(
          result, attr, dim_tensor1, dim_tensor2, result_shape);
      torch_ipex::xpu::oneDNN::quantized_matmul(
          result, input_view, weight, bias, false, attr);
      result = result.view(output_shape);
      return;
    }
  }

  bool is_fused() {
    return is_fused_;
  }

  bool is_fused_;
};

#define IPEX_QLINEAR_DEFINATION(func)                                      \
  Tensor q_linear_##func(                                                  \
      Tensor input,                                                        \
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,     \
      double output_scale,                                                 \
      int64_t output_zero_point) {                                         \
    RECORD_FUNCTION("q_linear_" #func, std::vector<c10::IValue>({input})); \
    auto q_linear_wrapper = QLinearConverter();                            \
    auto post_op = [=]() {                                                 \
      Attr attr(static_cast<float>(output_scale), output_zero_point);      \
      attr.append_post_eltwise(                                            \
          /* scale */ 1.f,                                                 \
          /* alpha */ 0.f,                                                 \
          /* beta */ 0.f,                                                  \
          attr.kind_with_##func);                                          \
      return attr;                                                         \
    };                                                                     \
    Tensor output;                                                         \
    q_linear_wrapper.call(                                                 \
        output,                                                            \
        input,                                                             \
        packed_weight,                                                     \
        output_scale,                                                      \
        output_zero_point,                                                 \
        input.scalar_type(),                                               \
        post_op);                                                          \
    return output;                                                         \
  }

Tensor q_linear_sigmoid(
    Tensor input,
    const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  RECORD_FUNCTION("q_linear_sigmoid", std::vector<c10::IValue>({input}));
  auto q_linear_wrapper = QLinearConverter();
  int64_t output_zp = (input.scalar_type() == at::kQInt8) ? -128 : 0;
  auto post_op = [=]() {
    Attr attr(static_cast<float>(1.0 / 256.0), output_zp);
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_sigmoid);
    return attr;
  };
  Tensor output;
  q_linear_wrapper.call(
      output,
      input,
      packed_weight,
      1.0 / 256.0,
      output_zp,
      input.scalar_type(),
      post_op);
  return output;
}

IPEX_QLINEAR_DEFINATION(relu)

at::Tensor QLinear(
    Tensor input,
    const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto q_linear_wrapper = QLinearConverter();
  auto post_op = [=]() {
    Attr attr(output_scale, output_zero_point);
    return attr;
  };
  Tensor output;
  q_linear_wrapper.call(
      output,
      input,
      packed_weight,
      output_scale,
      output_zero_point,
      input.scalar_type(),
      post_op);
  return output;
}

// result = (input * weight + bias + alpha * accumul)
// accumul = linear(input, weight, bias) + alpha * accumul
// inplace accumul tensor
Tensor q_linear_sum(
    Tensor input,
    const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Tensor& accumu) {
  RECORD_FUNCTION("q_linear_sum", std::vector<c10::IValue>({input}));
  auto post_op = [=]() {
    Attr attr(
        /* q_scale */ static_cast<float>(output_scale), output_zero_point);
    attr.append_post_eltwise(
            1,
            /*alpha*/ 1.0 / accumu.q_scale(),
            /*beta*/ 0.f,
            attr.kind_with_linear)
        .append_post_sum(1.f, 1.f)
        .append_post_eltwise(
            1,
            /*alpha*/ accumu.q_scale(),
            -accumu.q_zero_point() * accumu.q_scale(),
            attr.kind_with_linear);
    return attr;
  };
  auto q_linear_wrapper = QLinearConverter();
  q_linear_wrapper.call(
      accumu,
      input,
      packed_weight,
      output_scale,
      output_zero_point,
      input.scalar_type(),
      post_op);
  set_quantizer_(
      accumu,
      dpcpp_make_per_tensor_affine_quantizer(
          output_scale, output_zero_point, input.scalar_type()));

  return accumu;
}

Tensor q_linear_log_sigmoid(
    Tensor input,
    const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  RECORD_FUNCTION("q_linear_log_sigmoid", std::vector<c10::IValue>({input}));
  auto q_linear_wrapper = QLinearConverter();
  auto post_op = [=]() {
    Attr attr;
    attr.append_post_eltwise(
        /* scale */ 1.f,
        /* alpha */ -1.f,
        /* beta */ 0.f,
        attr.kind_with_soft_relu);
    return attr;
  };
  Tensor output;
  q_linear_wrapper.call(
      output,
      input,
      packed_weight,
      output_scale,
      output_zero_point,
      input.scalar_type(),
      post_op);
  return output;
}

#define IPEX_OP_REGISTER_QLINEAR(op) \
  IPEX_OP_REGISTER("q_linear_" #op, q_linear_##op);

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_QLINEAR(sigmoid);
  IPEX_OP_REGISTER_QLINEAR(relu);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  IPEX_QOP_REGISTER("quantized::linear", QLinear);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
