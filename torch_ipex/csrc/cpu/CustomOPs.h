#pragma once

#include "DevOPs.h"
#include "aten/aten.hpp"
#include "dbl/Common.h"
#include "dil/dil.hpp"
#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/utils.h"
#include "ShadeDataContext.h"
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>

class IPEXLayerNorm : public torch::autograd::Function<IPEXLayerNorm> {
public:
  static std::tuple<at::Tensor, at::Tensor, at::Tensor> _forward_impl(
      const at::Tensor & input,
      at::IntArrayRef normalized_shape,
      const c10::optional<at::Tensor> & weight,
      const c10::optional<at::Tensor> & bias,
      double eps) {
    try {
      if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::XPU) {
        return torch_ipex::cpu::AtenIpexCPUDev::dil_native_layer_norm(
          input,
          normalized_shape,
          weight,
          bias,
          eps);
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }

    return at::native_layer_norm(input, normalized_shape, weight, bias, eps);
  }

  static at::Tensor _forward(
      const at::Tensor & input,
      at::IntArrayRef normalized_shape,
      const c10::optional<at::Tensor> & weight,
      const c10::optional<at::Tensor> & bias,
      double eps) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXLayerNorm::_forward", std::vector<c10::IValue>({}));
#endif
    auto res = _forward_impl(input, normalized_shape, weight, bias, eps);
    return std::get<0>(res);
  }

  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      const at::Tensor & input,
      at::IntArrayRef normalized_shape,
      const c10::optional<at::Tensor> & weight,
      const c10::optional<at::Tensor> & bias,
      double eps) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXLayerNorm::forward", std::vector<c10::IValue>({}));
#endif
    at::AutoNonVariableTypeMode g;
    auto res = _forward_impl(input, normalized_shape, weight, bias, eps);
    auto mean = std::get<1>(res);
    auto rstd = std::get<2>(res);
    ctx->saved_data["normalized_shape"] = normalized_shape;
    ctx->save_for_backward({
      input,
      weight.has_value() ? weight.value() : at::Tensor(),
      bias.has_value() ? bias.value() : at::Tensor(),
      mean,
      rstd});

    return std::get<0>(res);
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXLayerNorm::backward", std::vector<c10::IValue>({}));
#endif
    auto normalized_shape = ctx->saved_data["normalized_shape"].toIntVector();
    auto saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];
    at::Tensor weight = saved[1];
    at::Tensor bias = saved[2];
    at::Tensor mean = saved[3];
    at::Tensor rstd = saved[4];

    at::Tensor grad_output = grad_outputs[0];

    try {
      if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::XPU) {
        auto res = torch_ipex::cpu::AtenIpexCPUDev::dil_native_layer_norm_backward(
          grad_output,
          input,
          normalized_shape,
          mean,
          rstd,
          weight,
          bias,
          {});
        return {std::get<0>(res), at::Tensor(), std::get<1>(res), std::get<2>(res), at::Tensor()};
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }

    auto res = at::native_layer_norm_backward(
      grad_output,
      input,
      normalized_shape,
      mean,
      rstd,
      weight,
      bias,
      {});
    return {std::get<0>(res), at::Tensor(), std::get<1>(res), std::get<2>(res), at::Tensor()};
  }
};

class NewLinearOp : public torch::autograd::Function<NewLinearOp> {
public:
  static at::Tensor _forward(at::Tensor input, at::Tensor weight,
                             at::Tensor bias = at::Tensor()) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXLinearOp::_forward",
                    std::vector<c10::IValue>({input, weight, bias}));
#endif
    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        return torch_ipex::cpu::AtenIpexCPUDev::dil_linear(input, weight, bias);
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    if (input.device().type() == c10::DeviceType::XPU) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == c10::kStrided);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bias.layout() == c10::kStrided);
      auto &&_ipex_input =
          torch_ipex::bridge::shallowFallbackToCPUTensor(input);
      auto &&_ipex_weight =
          torch_ipex::bridge::shallowFallbackToCPUTensor(weight);
      auto &&_ipex_bias = torch_ipex::bridge::shallowFallbackToCPUTensor(bias);
      auto &&_ipex_result = at::linear(_ipex_input, _ipex_weight, _ipex_bias);
      static_cast<void>(_ipex_result); // Avoid warnings in case not used
      return torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
    } else {
      return at::linear(input, weight, bias);
    }
  }

  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            at::Tensor input, at::Tensor weight,
                            at::Tensor bias = at::Tensor()) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXLinearOp::forward",
                    std::vector<c10::IValue>({input, weight, bias}));
#endif
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({input, weight, bias});
    return _forward(input, weight, bias);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXLinearOp::backward", std::vector<c10::IValue>({}));
#endif
    auto saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];
    at::Tensor weight = saved[1];
    at::Tensor bias = saved[2];

    at::Tensor grad_output = grad_outputs[0];
    at::Tensor grad_input, grad_weight;
    at::Tensor grad_bias = torch::Tensor();

    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        grad_input = torch_ipex::cpu::AtenIpexCPUDev::dil_linear_backward_input(
            input.sizes(),
            grad_output.is_contiguous() ? grad_output
                                        : grad_output.contiguous(),
            weight.is_contiguous() ? weight : weight.contiguous());
        std::tie(grad_weight, grad_bias) =
            torch_ipex::cpu::AtenIpexCPUDev::dil_linear_backward_weights(
                grad_output.is_contiguous() ? grad_output
                                            : grad_output.contiguous(),
                input.is_contiguous() ? input : input.contiguous(),
                weight.is_contiguous() ? weight : weight.contiguous(),
                bias.defined());
        return {grad_input, grad_weight, grad_bias};
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    if (input.dim() == 1) {
      grad_output = grad_output.unsqueeze(0);
      input = input.unsqueeze(0);
    } else if (input.dim() > 2) {
      std::vector<int64_t> mm_output_size;
      mm_output_size.push_back(-1);
      mm_output_size.push_back(weight.sizes()[0]);
      grad_output = grad_output.reshape(mm_output_size);
    }
    if (input.device().type() == c10::DeviceType::XPU) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == c10::kStrided);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_output.layout() == c10::kStrided);
      auto &&_ipex_input =
          torch_ipex::bridge::shallowFallbackToCPUTensor(input);
      auto &&_ipex_weight =
          torch_ipex::bridge::shallowFallbackToCPUTensor(weight);
      auto &&_ipex_grad_output =
          torch_ipex::bridge::shallowFallbackToCPUTensor(grad_output);
      if (input.dim() <= 2) {
        grad_input = _ipex_grad_output.mm(_ipex_weight);
        if (input.dim() == 1)
          grad_input = grad_input.squeeze_(0);
        grad_weight = _ipex_grad_output.t().mm(_ipex_input);
      } else { // input.dim() > 2
        grad_input = _ipex_grad_output.mm(_ipex_weight).reshape(input.sizes());
        grad_weight = _ipex_grad_output.t().mm(
            _ipex_input.view({-1, input.sizes()[input.dim() - 1]}));
      }
      if (bias.defined()) {
        grad_bias = _ipex_grad_output.sum(0);
      }
      static_cast<void>(grad_input);
      static_cast<void>(grad_weight);
      static_cast<void>(grad_bias);
      return {torch_ipex::bridge::shallowUpgradeToDPCPPTensor(grad_input),
              torch_ipex::bridge::shallowUpgradeToDPCPPTensor(grad_weight),
              torch_ipex::bridge::shallowUpgradeToDPCPPTensor(grad_bias)};
    } else {
      if (input.dim() <= 2) {
        grad_input = grad_output.mm(weight);
        if (input.dim() == 1)
          grad_input = grad_input.squeeze_(0);
        grad_weight = grad_output.t().mm(input);
      } else if (input.dim() > 2) {
        grad_input = grad_output.mm(weight).reshape(input.sizes());
        grad_weight = grad_output.t().mm(
            input.view({-1, input.sizes()[input.dim() - 1]}));
      }
      if (bias.defined()) {
        grad_bias = grad_output.sum(0);
      }
      return {grad_input, grad_weight, grad_bias};
    }
  }
};

class NewMaxPool2dOp : public torch::autograd::Function<NewMaxPool2dOp> {
public:
  static std::tuple<at::Tensor, at::Tensor>
  _forward(at::Tensor input, at::IntArrayRef kernel_size,
           at::IntArrayRef stride, at::IntArrayRef padding,
           at::IntArrayRef dilation, bool ceil_mode) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXMaxPool2dOp::_forward",
                    std::vector<c10::IValue>({input}));
#endif
    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        auto src_dil_type =
            torch_ipex::cpu::dbl::comm::try_gen_dil_tensor(input)
                .get_data_type();
        auto input_temp =
            (src_dil_type == dil::data_type::u8 ||
             src_dil_type == dil::data_type::s8 || input.is_contiguous())
                ? input
                : input.contiguous();

        at::Tensor output = torch_ipex::cpu::AtenIpexCPUDev::dil_max_pooling(
            input_temp, kernel_size, stride, padding, dilation, ceil_mode);
        return std::tuple<at::Tensor, at::Tensor>(output, output);
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    at::Tensor output, indices;
    if (input.device().type() == c10::DeviceType::XPU) {
      auto &&_ipex_input =
          torch_ipex::bridge::shallowFallbackToCPUTensor(input);
      auto &&_ipex_result = at::max_pool2d_with_indices(
          _ipex_input, kernel_size, stride, padding, dilation, ceil_mode);
      static_cast<void>(_ipex_result);
      std::tie(output, indices) = std::tuple<at::Tensor, at::Tensor>(
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              std::get<0>(_ipex_result)),
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              std::get<1>(_ipex_result)));
    } else {
      std::tie(output, indices) = at::max_pool2d_with_indices(
          input, kernel_size, stride, padding, dilation, ceil_mode);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, indices);
  }

  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            at::Tensor input, at::IntArrayRef kernel_size,
                            at::IntArrayRef stride, at::IntArrayRef padding,
                            at::IntArrayRef dilation, bool ceil_mode) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXMaxPool2dOp::forward",
                    std::vector<c10::IValue>({input}));
#endif
    ctx->saved_data["kernel_size"] = kernel_size;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["ceil_mode"] = ceil_mode;

    at::Tensor output, indices;
    std::tie(output, indices) =
        _forward(input, kernel_size, stride, padding, dilation, ceil_mode);
    ctx->save_for_backward({input, indices});
    return output;
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXMaxPool2dOp::backward", std::vector<c10::IValue>({}));
#endif
    auto saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];
    at::Tensor indices = saved[1];

    at::Tensor grad_output = grad_outputs[0];
    at::Tensor grad_input;

    std::vector<int64_t> kernel_size =
        ctx->saved_data["kernel_size"].toIntVector();
    std::vector<int64_t> stride = ctx->saved_data["stride"].toIntVector();
    std::vector<int64_t> padding = ctx->saved_data["padding"].toIntVector();
    std::vector<int64_t> dilation = ctx->saved_data["dilation"].toIntVector();
    bool ceil_mode = ctx->saved_data["ceil_mode"].toBool();

    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        grad_input = torch_ipex::cpu::AtenIpexCPUDev::dil_max_pooling_backward(
            grad_output.is_contiguous() ? grad_output
                                        : grad_output.contiguous(),
            indices.is_contiguous() ? indices : indices.contiguous(),
            input.is_contiguous() ? input : input.contiguous(), kernel_size,
            stride, padding, dilation, ceil_mode);
        return {grad_input,   at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor()};
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    if (input.device().type() == c10::DeviceType::XPU) {
      auto &&_ipex_grad_output =
          torch_ipex::bridge::shallowFallbackToCPUTensor(grad_output);
      auto &&_ipex_input =
          torch_ipex::bridge::shallowFallbackToCPUTensor(input);
      auto &&_ipex_indices =
          torch_ipex::bridge::shallowFallbackToCPUTensor(indices);
      auto &&_ipex_grad_input = at::max_pool2d_with_indices_backward(
          _ipex_grad_output, _ipex_input, kernel_size, stride, padding,
          dilation, ceil_mode, _ipex_indices);
      static_cast<void>(_ipex_grad_input);
      grad_input =
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_grad_input);
    } else {
      grad_input = at::max_pool2d_with_indices_backward(
          grad_output, input, kernel_size, stride, padding, dilation, ceil_mode,
          indices);
    }
    return {grad_input,   at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class NewMaxPool3dOp : public torch::autograd::Function<NewMaxPool3dOp> {
public:
  static std::tuple<at::Tensor, at::Tensor>
  _forward(at::Tensor input, at::IntArrayRef kernel_size,
           at::IntArrayRef stride, at::IntArrayRef padding,
           at::IntArrayRef dilation, bool ceil_mode) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXMaxPool3dOp::_forward",
                    std::vector<c10::IValue>({input}));
#endif
    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        at::Tensor output = torch_ipex::cpu::AtenIpexCPUDev::dil_max_pooling(
            input.is_contiguous() ? input : input.contiguous(), kernel_size,
            stride, padding, dilation, ceil_mode);
        return std::tuple<at::Tensor, at::Tensor>(output, output);
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    at::Tensor output, indices;
    if (input.device().type() == c10::DeviceType::XPU) {
      auto &&_ipex_input =
          torch_ipex::bridge::shallowFallbackToCPUTensor(input);
      auto &&_ipex_result = at::max_pool3d_with_indices(
          _ipex_input, kernel_size, stride, padding, dilation, ceil_mode);
      static_cast<void>(_ipex_result);
      std::tie(output, indices) = std::tuple<at::Tensor, at::Tensor>(
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              std::get<0>(_ipex_result)),
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              std::get<1>(_ipex_result)));
    } else {
      std::tie(output, indices) = at::max_pool3d_with_indices(
          input, kernel_size, stride, padding, dilation, ceil_mode);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, indices);
  }

  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            at::Tensor input, at::IntArrayRef kernel_size,
                            at::IntArrayRef stride, at::IntArrayRef padding,
                            at::IntArrayRef dilation, bool ceil_mode) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXMaxPool3dOp::forward",
                    std::vector<c10::IValue>({input}));
#endif
    ctx->saved_data["kernel_size"] = kernel_size;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["ceil_mode"] = ceil_mode;

    at::Tensor output, indices;
    std::tie(output, indices) =
        _forward(input, kernel_size, stride, padding, dilation, ceil_mode);
    ctx->save_for_backward({input, indices});
    return output;
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXMaxPool3dOp::backward", std::vector<c10::IValue>({}));
#endif
    auto saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];
    at::Tensor indices = saved[1];

    at::Tensor grad_output = grad_outputs[0];
    at::Tensor grad_input;

    std::vector<int64_t> kernel_size =
        ctx->saved_data["kernel_size"].toIntVector();
    std::vector<int64_t> stride = ctx->saved_data["stride"].toIntVector();
    std::vector<int64_t> padding = ctx->saved_data["padding"].toIntVector();
    std::vector<int64_t> dilation = ctx->saved_data["dilation"].toIntVector();
    bool ceil_mode = ctx->saved_data["ceil_mode"].toBool();

    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        grad_input = torch_ipex::cpu::AtenIpexCPUDev::dil_max_pooling_backward(
            grad_output.is_contiguous() ? grad_output
                                        : grad_output.contiguous(),
            indices.is_contiguous() ? indices : indices.contiguous(),
            input.is_contiguous() ? input : input.contiguous(), kernel_size,
            stride, padding, dilation, ceil_mode);
        return {grad_input,   at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor()};
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    if (input.device().type() == c10::DeviceType::XPU) {
      auto &&_ipex_grad_output =
          torch_ipex::bridge::shallowFallbackToCPUTensor(grad_output);
      auto &&_ipex_input =
          torch_ipex::bridge::shallowFallbackToCPUTensor(input);
      auto &&_ipex_indices =
          torch_ipex::bridge::shallowFallbackToCPUTensor(indices);
      auto &&_ipex_grad_input = at::max_pool3d_with_indices_backward(
          _ipex_grad_output, _ipex_input, kernel_size, stride, padding,
          dilation, ceil_mode, _ipex_indices);
      static_cast<void>(_ipex_grad_input);
      grad_input =
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_grad_input);
    } else {
      grad_input = at::max_pool3d_with_indices_backward(
          grad_output, input, kernel_size, stride, padding, dilation, ceil_mode,
          indices);
    }
    return {grad_input,   at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class NewApaptiveAvgPoolingOp
    : public torch::autograd::Function<NewApaptiveAvgPoolingOp> {
public:
  static at::Tensor _forward(at::Tensor input, at::IntArrayRef output_size) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXApaptiveAvgPoolingOp::_forward",
                    std::vector<c10::IValue>({input}));
#endif
    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        auto src_dil_type =
            torch_ipex::cpu::dbl::comm::try_gen_dil_tensor(input)
                .get_data_type();
        auto input_temp =
            (src_dil_type == dil::data_type::u8 ||
             src_dil_type == dil::data_type::s8 || input.is_contiguous())
                ? input
                : input.contiguous();
        return torch_ipex::cpu::AtenIpexCPUDev::dil_adaptive_avg_pool2d(
            input_temp, output_size);
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    if (input.device().type() == c10::DeviceType::XPU) {
      auto &&_ipex_input =
          torch_ipex::bridge::shallowFallbackToCPUTensor(input);
      auto &&_ipex_result = at::_adaptive_avg_pool2d(_ipex_input, output_size);
      static_cast<void>(_ipex_result); // Avoid warnings in case not used
      return torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
    } else {
      return at::_adaptive_avg_pool2d(input, output_size);
    }
  }

  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            at::Tensor input, at::IntArrayRef output_size) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXApaptiveAvgPoolingOp::forward",
                    std::vector<c10::IValue>({input}));
#endif
    ctx->save_for_backward({input});
    return _forward(input, output_size);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXApaptiveAvgPoolingOp::backward",
                    std::vector<c10::IValue>({}));
#endif
    auto saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];

    at::Tensor grad_output = grad_outputs[0];
    at::Tensor grad_input;

    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        grad_input =
            torch_ipex::cpu::AtenIpexCPUDev::dil_adaptive_avg_pool2d_backward(
                grad_output.is_contiguous() ? grad_output
                                            : grad_output.contiguous(),
                input.is_contiguous() ? input : input.contiguous());
        return {grad_input, at::Tensor()};
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    if (input.device().type() == c10::DeviceType::XPU) {
      auto &&_ipex_grad_output =
          torch_ipex::bridge::shallowFallbackToCPUTensor(grad_output);
      auto &&_ipex_input =
          torch_ipex::bridge::shallowFallbackToCPUTensor(input);
      auto &&_ipex_result =
          at::_adaptive_avg_pool2d_backward(_ipex_grad_output, _ipex_input);
      static_cast<void>(_ipex_result); // Avoid warnings in case not used
      grad_input =
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
    } else {
      grad_input = at::_adaptive_avg_pool2d_backward(grad_output, input);
    }
    return {grad_input, at::Tensor()};
  }
};

class NewEmbeddingBagOp : public torch::autograd::Function<NewEmbeddingBagOp> {
public:
  static std::vector<at::Tensor>
  _forward(const at::Tensor &weight, const at::Tensor &indices,
           const at::Tensor &offsets, bool scale_grad_by_freq, int64_t mode,
           bool sparse, bool include_last_offset,
           const at::Tensor per_sample_weights = at::Tensor()) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXEmbeddingBagOp::_forward",
                    std::vector<c10::IValue>({weight, indices, offsets}));
#endif
    try {
      if (torch_ipex::check_auto_dnnl() &&
          torch_ipex::cpu::aten::embedding_bag::embedding_bag_fast_path_sum(
              weight, per_sample_weights, mode) &&
          weight.device().type() == c10::DeviceType::XPU &&
          indices.device().type() == c10::DeviceType::XPU &&
          offsets.device().type() == c10::DeviceType::XPU) {
        auto ret = torch_ipex::cpu::aten::embedding_bag::embedding_bag_impl(
            weight, indices, offsets, scale_grad_by_freq, mode, sparse,
            per_sample_weights, include_last_offset);
        return {ret};
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    if (torch_ipex::check_auto_dnnl() &&
        weight.device().type() == c10::DeviceType::XPU &&
        indices.device().type() == c10::DeviceType::XPU &&
        offsets.device().type() == c10::DeviceType::XPU) {
      auto &&_ipex_weight =
          torch_ipex::bridge::shallowFallbackToCPUTensor(weight);
      auto &&_ipex_indices =
          torch_ipex::bridge::shallowFallbackToCPUTensor(indices);
      auto &&_ipex_offsets =
          torch_ipex::bridge::shallowFallbackToCPUTensor(offsets);
      auto &&_ipex_per_sample_weights =
          torch_ipex::bridge::shallowFallbackToCPUTensor(per_sample_weights);
      auto &&_ipex_result = at::embedding_bag(
          _ipex_weight, _ipex_indices, _ipex_offsets, scale_grad_by_freq, mode,
          sparse, _ipex_per_sample_weights, include_last_offset);
      static_cast<void>(_ipex_result); // Avoid warnings in case not used
      return {
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              std::get<0>(_ipex_result)),
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              std::get<1>(_ipex_result)),
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              std::get<2>(_ipex_result)),
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              std::get<3>(_ipex_result))};
    } else {
      auto ret =
          at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode,
                            sparse, per_sample_weights, include_last_offset);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret),std::get<3>(ret)};
    }
  }

  static std::vector<at::Tensor>
  forward(torch::autograd::AutogradContext *ctx, const at::Tensor &weight,
          const at::Tensor &indices, const at::Tensor &offsets,
          bool scale_grad_by_freq, int64_t mode, bool sparse,
          bool include_last_offset,
          const at::Tensor per_sample_weights = at::Tensor()) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXEmbeddingBagOp::forward",
                    std::vector<c10::IValue>({weight, indices, offsets}));
#endif
    at::AutoNonVariableTypeMode g;
    ctx->saved_data["num_weights"] = weight.size(0);
    ctx->saved_data["scale_grad_by_freq"] = scale_grad_by_freq;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["sparse"] = sparse;
    ctx->saved_data["include_last_offset"] = include_last_offset;
    auto ret = _forward(weight, indices, offsets, scale_grad_by_freq, mode,
                        sparse, include_last_offset, per_sample_weights);
    if (ret.size() != 1)
      ctx->save_for_backward({weight, indices, offsets, per_sample_weights,
                              ret[1], ret[2],
                              ret[3]});
    else
      ctx->save_for_backward({weight, indices, offsets, per_sample_weights,
                              at::empty({0}, offsets.options()), at::Tensor(),
                              at::Tensor()});
    return ret;
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("IPEXEmbeddingBagOp::backward", std::vector<c10::IValue>({}));
#endif
    at::AutoNonVariableTypeMode g;
    auto saved = ctx->get_saved_variables();
    at::Tensor weight = saved[0];
    at::Tensor indices = saved[1];
    at::Tensor offsets = saved[2];
    at::Tensor per_sample_weights = saved[3];
    at::Tensor offset2bag = saved[4];
    at::Tensor bag_size = saved[5];
    at::Tensor maximum_indices = saved[6];

    int64_t num_weights = ctx->saved_data["num_weights"].toInt();
    bool scale_grad_by_freq = ctx->saved_data["scale_grad_by_freq"].toBool();
    int64_t mode = ctx->saved_data["mode"].toInt();
    bool sparse = ctx->saved_data["sparse"].toBool();
    bool include_last_offset = ctx->saved_data["include_last_offset"].toBool();

    at::Tensor grad = grad_outputs[0];
    if (!sparse)
      grad = grad.contiguous();

    try {
      if (torch_ipex::check_auto_dnnl() &&
          (torch_ipex::cpu::aten::embedding_bag::
               embedding_bag_backward_fast_path_sum(
                   grad, indices, offset2bag, per_sample_weights,
                   scale_grad_by_freq, mode)) &&
          weight.device().type() == c10::DeviceType::XPU &&
          indices.device().type() == c10::DeviceType::XPU &&
          offsets.device().type() == c10::DeviceType::XPU) {
        return {
            torch_ipex::cpu::aten::embedding_bag::embedding_bag_backward_impl(
                grad, indices, offsets, offset2bag, bag_size, maximum_indices,
                num_weights, scale_grad_by_freq, mode, sparse,
                per_sample_weights),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor()};
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    bool compute_per_sample_weights_grad =
        per_sample_weights.defined() && per_sample_weights.requires_grad();
    at::Tensor offset2bag_ =
        torch_ipex::cpu::aten::embedding_bag::embedding_bag_get_offset2bag(
            indices, offsets, offset2bag);
    if (torch_ipex::check_auto_dnnl() &&
        grad.device().type() == c10::DeviceType::XPU &&
        indices.device().type() == c10::DeviceType::XPU &&
        offsets.device().type() == c10::DeviceType::XPU &&
        (!per_sample_weights.defined() ||
         per_sample_weights.device().type() == c10::DeviceType::XPU)) {
      auto &&_ipex_grad = torch_ipex::bridge::shallowFallbackToCPUTensor(grad);
      auto &&_ipex_weight =
          torch_ipex::bridge::shallowFallbackToCPUTensor(weight);
      auto &&_ipex_indices =
          torch_ipex::bridge::shallowFallbackToCPUTensor(indices);
      auto &&_ipex_offsets =
          torch_ipex::bridge::shallowFallbackToCPUTensor(offsets);
      auto &&_ipex_offset2bag_ =
          torch_ipex::bridge::shallowFallbackToCPUTensor(offset2bag_);
      auto &&_ipex_bag_size =
          torch_ipex::bridge::shallowFallbackToCPUTensor(bag_size);
      auto &&_ipex_maximum_indices =
          torch_ipex::bridge::shallowFallbackToCPUTensor(maximum_indices);
      auto &&_ipex_per_sample_weights =
          torch_ipex::bridge::shallowFallbackToCPUTensor(per_sample_weights);
      auto &&_ipex_weight_grad =
          sparse
              ? at::_embedding_bag_sparse_backward(
                    _ipex_grad, _ipex_indices, _ipex_offsets, _ipex_offset2bag_,
                    _ipex_bag_size, num_weights, scale_grad_by_freq, mode,
                    _ipex_per_sample_weights)
              : at::_embedding_bag_dense_backward(
                    _ipex_grad, _ipex_indices, _ipex_offsets, _ipex_offset2bag_,
                    _ipex_bag_size, _ipex_maximum_indices, num_weights,
                    scale_grad_by_freq, mode, _ipex_per_sample_weights);
      auto &&_ipex_per_sample_weights_grad =
          compute_per_sample_weights_grad
              ? at::_embedding_bag_per_sample_weights_backward(
                    _ipex_grad, _ipex_weight, _ipex_indices, _ipex_offsets,
                    _ipex_offset2bag_, mode)
              : at::Tensor();
      auto per_sample_weights_grad =
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(
              _ipex_per_sample_weights_grad);
      return {
          torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_weight_grad),
          at::Tensor(),
          at::Tensor(),
          at::Tensor(),
          at::Tensor(),
          at::Tensor(),
          at::Tensor(),
          per_sample_weights_grad};
    } else {
      auto weight_grad =
          sparse
              ? at::_embedding_bag_sparse_backward(
                    grad, indices, offsets, offset2bag_, bag_size, num_weights,
                    scale_grad_by_freq, mode, per_sample_weights)
              : at::_embedding_bag_dense_backward(
                    grad, indices, offsets, offset2bag_, bag_size,
                    maximum_indices, num_weights, scale_grad_by_freq, mode,
                    per_sample_weights);
      auto per_sample_weights_grad =
          compute_per_sample_weights_grad
              ? at::_embedding_bag_per_sample_weights_backward(
                    grad, grad, indices, offsets, offset2bag_, mode)
              : at::Tensor();
      return {weight_grad,  per_sample_weights_grad,
              at::Tensor(), at::Tensor(),
              at::Tensor(), at::Tensor(),
              at::Tensor(), at::Tensor()};
    }
  }
};

class NewRNNLayerOp : public torch::autograd::Function<NewRNNLayerOp> {
public:
  static std::vector<at::Tensor> _forward(const at::Tensor& input, const at::Tensor& w1, const at::Tensor& w2,
    const at::Tensor& w3, const at::Tensor& w4, const at::Tensor& hx, const at::Tensor& cx, bool reverse, int64_t mode,
    int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("NewRNNLayerOp::_forward", std::vector<c10::IValue>({input, w1, w2, w3, w4, hx, cx, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        return torch_ipex::cpu::AtenIpexCPUDev::dil_rnn_layer(
            input, w1, w2, w3, w4, hx, cx, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes);
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    IPEX_CHECK(false, "RNN forward not support fallback path now");
  }

  static std::vector<at::Tensor> forward(torch::autograd::AutogradContext *ctx, const at::Tensor& input, const at::Tensor& w1,
    const at::Tensor& w2, const at::Tensor& w3, const at::Tensor& w4, const at::Tensor& hx, const at::Tensor& cx, bool reverse,
    int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes) {
    ctx->saved_data["reverse"] = reverse;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["hidden_size"] = hidden_size;
    ctx->saved_data["num_layers"] = num_layers;
    ctx->saved_data["has_biases"] = has_biases;
    ctx->saved_data["train"] = train;
    ctx->saved_data["bidirectional"] = bidirectional;
    auto outputs = _forward(input, w1, w2, w3, w4, hx, cx, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes);
    ctx->save_for_backward({input, w1, w2, w3, w4, hx, cx, outputs[0], outputs[1], outputs[2]});
    return outputs;
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("NewRNNLayerOp::backward", std::vector<c10::IValue>({}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
    auto saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];
    at::Tensor w1 = saved[1];
    at::Tensor w2 = saved[2];
    at::Tensor w3 = saved[3];
    at::Tensor w4 = saved[4];
    at::Tensor hx = saved[5];
    at::Tensor cx = saved[6];
    at::Tensor output = saved[7];
    at::Tensor hy = saved[8];
    at::Tensor cy = saved[9];
    bool reverse = ctx->saved_data["reverse"].toBool();
    int64_t mode = ctx->saved_data["mode"].toInt();
    int64_t hidden_size = ctx->saved_data["hidden_size"].toInt();
    int64_t num_layers = ctx->saved_data["num_layers"].toInt();
    bool has_biases = ctx->saved_data["has_biases"].toBool();
    bool train = ctx->saved_data["train"].toBool();
    bool bidirectional = ctx->saved_data["bidirectional"].toBool();
    at::Tensor grad_output = grad_outputs[0].contiguous();
    at::Tensor grad_hy = grad_outputs[1].contiguous();
    at::Tensor grad_cy = grad_outputs[2].contiguous();

    try {
      if (torch_ipex::check_auto_dnnl() &&
          input.device().type() == c10::DeviceType::XPU) {
        auto grad_inputs = torch_ipex::cpu::AtenIpexCPUDev::dil_rnn_layer_backward(input, w1, w2, w3, w4, hx, cx, output, hy, cy, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, /*batch_sizes*/{});
        return {grad_inputs[0], grad_inputs[1], grad_inputs[2],
          grad_inputs[3], grad_inputs[4], grad_inputs[5],
          grad_inputs[6], at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor()};
      }
    } catch (std::exception &e) {
#if defined(_DEBUG)
      TORCH_WARN(e.what());
#endif
    }
    IPEX_CHECK(false, "RNN backward not support fallback path now");
  }
};

class FrozenBatchNormOp : public torch::autograd::Function<FrozenBatchNormOp> {
public:
  static at::Tensor _forward(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean, const at::Tensor& running_var) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("FrozenBatchNormOp::_forward",
                    std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}),
                    torch::autograd::Node::peek_at_next_sequence_nr());
#endif
    return torch_ipex::cpu::AtenIpexCPUDev::dil_frozen_batch_norm(input, weight, bias, running_mean, running_var, 0);
  }

  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean, const at::Tensor& running_var) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("FrozenBatchNormOp::forward",
                    std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}),
                    torch::autograd::Node::peek_at_next_sequence_nr());
#endif
    ctx->save_for_backward({input, weight, running_mean, running_var});
    return torch_ipex::cpu::AtenIpexCPUDev::dil_frozen_batch_norm(input, weight, bias, running_mean, running_var, 0);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
#if defined(IPEX_PROFILE_OP)
    RECORD_FUNCTION("FrozenBatchNormOp::backward", std::vector<c10::IValue>({}),
                    torch::autograd::Node::peek_at_next_sequence_nr());
#endif
    auto saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];
    at::Tensor weight = saved[1];
    at::Tensor running_mean = saved[2];
    at::Tensor running_var = saved[3];
    at::Tensor grad_output = grad_outputs[0].contiguous();
    auto output = torch_ipex::cpu::AtenIpexCPUDev::dil_frozen_batch_norm_backward(grad_output, input, weight, running_mean, running_var, 0);
    return {output, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};
