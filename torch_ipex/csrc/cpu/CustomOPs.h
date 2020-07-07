#pragma once

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/Tensor.h>
#include <torch/script.h>
#include <c10/util/Optional.h>
#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/utils.h"
#include "DevOPs.h"

class NewLinearOp : public torch::autograd::Function<NewLinearOp> {
  public:
      static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        at::Tensor weight,
        at::Tensor bias = at::Tensor()) {
        ctx->save_for_backward({input, weight, bias});
        try {
          if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
            return torch_ipex::cpu::AtenIpexCPUDev::dil_linear(input.is_contiguous() ? input : input.contiguous(), weight.is_contiguous() ? weight : weight.contiguous(), bias.is_contiguous() ? bias : bias.contiguous());
          } 
        } catch (std::exception& e) {
#if defined(_DEBUG)
          TORCH_WARN(e.what());
#endif
        }
        if (input.device().type() == c10::DeviceType::DPCPP) {
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == c10::kStrided);
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bias.layout() == c10::kStrided);
          auto&& _ipex_input = torch_ipex::bridge::shallowFallbackToCPUTensor(input);
          auto&& _ipex_weight = torch_ipex::bridge::shallowFallbackToCPUTensor(weight);
          auto&& _ipex_bias = torch_ipex::bridge::shallowFallbackToCPUTensor(bias);
          auto&& _ipex_result = at::linear(_ipex_input, _ipex_weight, _ipex_bias);
          static_cast<void>(_ipex_result); // Avoid warnings in case not used
          return torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
        } else {
          return at::linear(input, weight, bias);
        }
      }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
      auto saved = ctx->get_saved_variables();
      at::Tensor input = saved[0];
      at::Tensor weight = saved[1];
      at::Tensor bias = saved[2];

      at::Tensor grad_output = grad_outputs[0];
      at::Tensor grad_input, grad_weight;
      at::Tensor grad_bias = torch::Tensor();

      try {
        if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
          grad_input = torch_ipex::cpu::AtenIpexCPUDev::dil_linear_backward_input(
              input.sizes(), grad_output.is_contiguous() ? grad_output : grad_output.contiguous(), weight.is_contiguous() ? weight : weight.contiguous());
          std::tie(grad_weight, grad_bias) = torch_ipex::cpu::AtenIpexCPUDev::dil_linear_backward_weights(
              grad_output.is_contiguous() ? grad_output : grad_output.contiguous(), input.is_contiguous() ? input : input.contiguous(), weight.is_contiguous() ? weight : weight.contiguous(), bias.defined());
          return {grad_input, grad_weight, grad_bias};
        } 
      } catch (std::exception& e) {
#if defined(_DEBUG)
          TORCH_WARN(e.what());
#endif
      }
      if (input.device().type() == c10::DeviceType::DPCPP) {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == c10::kStrided);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_output.layout() == c10::kStrided);
        auto&& _ipex_input = torch_ipex::bridge::shallowFallbackToCPUTensor(input);
        auto&& _ipex_weight = torch_ipex::bridge::shallowFallbackToCPUTensor(weight);
        auto&& _ipex_grad_output = torch_ipex::bridge::shallowFallbackToCPUTensor(grad_output);
        grad_input = _ipex_grad_output.mm(_ipex_weight);
        grad_weight = _ipex_grad_output.t().mm(_ipex_input);
        if (bias.defined()) {
          grad_bias = _ipex_grad_output.sum(0);
        }
        static_cast<void>(grad_input);
        static_cast<void>(grad_weight);
        static_cast<void>(grad_bias);
        return {torch_ipex::bridge::shallowUpgradeToDPCPPTensor(grad_input), torch_ipex::bridge::shallowUpgradeToDPCPPTensor(grad_weight), torch_ipex::bridge::shallowUpgradeToDPCPPTensor(grad_bias)};
      } else {
        grad_input = grad_output.mm(weight);
        grad_weight = grad_output.t().mm(input);
        if (bias.defined()) {
          grad_bias = grad_output.sum(0);
        }
        return {grad_input, grad_weight, grad_bias};
      }
    }
};

class NewMaxPool2dOp : public torch::autograd::Function<NewMaxPool2dOp> {
  public:
      static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        at::IntArrayRef kernel_size,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation,
        bool ceil_mode) {
        ctx->saved_data["kernel_size"] = kernel_size;
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["ceil_mode"] = ceil_mode;

        try {
          if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
            at::Tensor output = torch_ipex::cpu::AtenIpexCPUDev::dil_max_pooling(input.is_contiguous() ? input : input.contiguous(), kernel_size, stride,
                padding, dilation, ceil_mode);
            ctx->save_for_backward({input, output});
            return output;
          }
        } catch (std::exception& e) {
#if defined(_DEBUG)
          TORCH_WARN(e.what());
#endif
        }
        at::Tensor output, indices;
        if (input.device().type() == c10::DeviceType::DPCPP) {
          auto&& _ipex_input = torch_ipex::bridge::shallowFallbackToCPUTensor(input);
          auto&& _ipex_result = at::max_pool2d_with_indices(_ipex_input, kernel_size, stride, padding, dilation, ceil_mode);
          static_cast<void>(_ipex_result);
          std::tie(output, indices) = std::tuple<at::Tensor,at::Tensor>(torch_ipex::bridge::shallowUpgradeToDPCPPTensor(std::get<0>(_ipex_result)), torch_ipex::bridge::shallowUpgradeToDPCPPTensor(std::get<1>(_ipex_result)));
        } else {
          std::tie(output, indices) = at::max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode);
        }
        ctx->save_for_backward({input, indices});
        return output;
      }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
      auto saved = ctx->get_saved_variables();
      at::Tensor input = saved[0];
      at::Tensor indices = saved[1];

      at::Tensor grad_output = grad_outputs[0];
      at::Tensor grad_input;

      std::vector<int64_t> kernel_size = ctx->saved_data["kernel_size"].toIntVector();
      std::vector<int64_t> stride = ctx->saved_data["stride"].toIntVector();
      std::vector<int64_t> padding = ctx->saved_data["padding"].toIntVector();
      std::vector<int64_t> dilation = ctx->saved_data["dilation"].toIntVector();
      bool ceil_mode = ctx->saved_data["ceil_mode"].toBool();

      
      try {
        if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
          grad_input = torch_ipex::cpu::AtenIpexCPUDev::dil_max_pooling_backward(
              grad_output.is_contiguous() ? grad_output : grad_output.contiguous(), indices.is_contiguous() ? indices : indices.contiguous(), input.is_contiguous() ? input : input.contiguous(), kernel_size, stride, padding, dilation, ceil_mode);
          return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }
      } catch (std::exception& e) {
#if defined(_DEBUG)
        TORCH_WARN(e.what());
#endif
      }
      if (input.device().type() == c10::DeviceType::DPCPP) {
        auto&& _ipex_grad_output = torch_ipex::bridge::shallowFallbackToCPUTensor(grad_output);
        auto&& _ipex_input = torch_ipex::bridge::shallowFallbackToCPUTensor(input);
        auto&& _ipex_indices = torch_ipex::bridge::shallowFallbackToCPUTensor(indices);
        auto&& _ipex_grad_input = at::max_pool2d_with_indices_backward(_ipex_grad_output, _ipex_input, kernel_size,
            stride, padding, dilation, ceil_mode, _ipex_indices);
        static_cast<void>(_ipex_grad_input);
        grad_input = torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_grad_input);
      } else {
        grad_input = at::max_pool2d_with_indices_backward(grad_output, input, kernel_size,
            stride, padding, dilation, ceil_mode, indices);
      }
      return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class NewMaxPool3dOp : public torch::autograd::Function<NewMaxPool3dOp> {
  public:
      static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        at::IntArrayRef kernel_size,
        at::IntArrayRef stride,
        at::IntArrayRef padding,
        at::IntArrayRef dilation,
        bool ceil_mode) {
        ctx->saved_data["kernel_size"] = kernel_size;
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["ceil_mode"] = ceil_mode;

        try {
          if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
            at::Tensor output = torch_ipex::cpu::AtenIpexCPUDev::dil_max_pooling(input.is_contiguous() ? input : input.contiguous(), kernel_size, stride,
                padding, dilation, ceil_mode);
            ctx->save_for_backward({input, output});
            return output;
          }
        } catch (std::exception& e) {
#if defined(_DEBUG)
          TORCH_WARN(e.what());
#endif
        }
        at::Tensor output, indices;
        if (input.device().type() == c10::DeviceType::DPCPP) {
          auto&& _ipex_input = torch_ipex::bridge::shallowFallbackToCPUTensor(input);
          auto&& _ipex_result = at::max_pool3d_with_indices(_ipex_input, kernel_size, stride, padding, dilation, ceil_mode);
          static_cast<void>(_ipex_result);
          std::tie(output, indices) = std::tuple<at::Tensor,at::Tensor>(torch_ipex::bridge::shallowUpgradeToDPCPPTensor(std::get<0>(_ipex_result)), torch_ipex::bridge::shallowUpgradeToDPCPPTensor(std::get<1>(_ipex_result)));
        } else {
          std::tie(output, indices) = at::max_pool3d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode);
        }
        ctx->save_for_backward({input, indices});
        return output;
      }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
      auto saved = ctx->get_saved_variables();
      at::Tensor input = saved[0];
      at::Tensor indices = saved[1];

      at::Tensor grad_output = grad_outputs[0];
      at::Tensor grad_input;

      std::vector<int64_t> kernel_size = ctx->saved_data["kernel_size"].toIntVector();
      std::vector<int64_t> stride = ctx->saved_data["stride"].toIntVector();
      std::vector<int64_t> padding = ctx->saved_data["padding"].toIntVector();
      std::vector<int64_t> dilation = ctx->saved_data["dilation"].toIntVector();
      bool ceil_mode = ctx->saved_data["ceil_mode"].toBool();

      
      try {
        if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
          grad_input = torch_ipex::cpu::AtenIpexCPUDev::dil_max_pooling_backward(
              grad_output.is_contiguous() ? grad_output : grad_output.contiguous(), indices.is_contiguous() ? indices : indices.contiguous(), input.is_contiguous() ? input : input.contiguous(), kernel_size, stride, padding, dilation, ceil_mode);
          return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }
      } catch (std::exception& e) {
#if defined(_DEBUG)
        TORCH_WARN(e.what());
#endif
      }
      if (input.device().type() == c10::DeviceType::DPCPP) {
        auto&& _ipex_grad_output = torch_ipex::bridge::shallowFallbackToCPUTensor(grad_output);
        auto&& _ipex_input = torch_ipex::bridge::shallowFallbackToCPUTensor(input);
        auto&& _ipex_indices = torch_ipex::bridge::shallowFallbackToCPUTensor(indices);
        auto&& _ipex_grad_input = at::max_pool3d_with_indices_backward(_ipex_grad_output, _ipex_input, kernel_size,
            stride, padding, dilation, ceil_mode, _ipex_indices);
        static_cast<void>(_ipex_grad_input);
        grad_input = torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_grad_input);
      } else {
        grad_input = at::max_pool3d_with_indices_backward(grad_output, input, kernel_size,
            stride, padding, dilation, ceil_mode, indices);
      }
      return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class NewApaptiveAvgPoolingOp : public torch::autograd::Function<NewApaptiveAvgPoolingOp> {
  public:
      static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        at::IntArrayRef output_size) {
        ctx->save_for_backward({input});

        try{
          if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
            return torch_ipex::cpu::AtenIpexCPUDev::dil_adaptive_avg_pool2d(input.is_contiguous() ? input : input.contiguous(), output_size);
          } 
        } catch (std::exception& e) {
#if defined(_DEBUG)
          TORCH_WARN(e.what());
#endif
        }
        if (input.device().type() == c10::DeviceType::DPCPP) {
          auto&& _ipex_input = torch_ipex::bridge::shallowFallbackToCPUTensor(input);
          auto&& _ipex_result = at::_adaptive_avg_pool2d(_ipex_input, output_size);
          static_cast<void>(_ipex_result); // Avoid warnings in case not used
          return torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
        } else {
          return at::_adaptive_avg_pool2d(input, output_size);
        }
      }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
      auto saved = ctx->get_saved_variables();
      at::Tensor input = saved[0];

      at::Tensor grad_output = grad_outputs[0];
      at::Tensor grad_input;

      try {
        if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
          grad_input = torch_ipex::cpu::AtenIpexCPUDev::dil_adaptive_avg_pool2d_backward(grad_output.is_contiguous() ? grad_output : grad_output.contiguous(), input.is_contiguous() ? input : input.contiguous());
          return {grad_input, at::Tensor()};
        }
      } catch (std::exception& e) {
#if defined(_DEBUG)
        TORCH_WARN(e.what());
#endif
      }
      if (input.device().type() == c10::DeviceType::DPCPP) {
        auto&& _ipex_grad_output = torch_ipex::bridge::shallowFallbackToCPUTensor(grad_output);
        auto&& _ipex_input = torch_ipex::bridge::shallowFallbackToCPUTensor(input);
        auto&& _ipex_result = at::_adaptive_avg_pool2d_backward(_ipex_grad_output, _ipex_input);
        static_cast<void>(_ipex_result); // Avoid warnings in case not used
        grad_input = torch_ipex::bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
      } else {
        grad_input = at::_adaptive_avg_pool2d_backward(grad_output, input);
      }
      return {grad_input, at::Tensor()};
    }
};
