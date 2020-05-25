#pragma once

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/Tensor.h>
#include <torch/script.h>
#include <c10/util/Optional.h>
#include "torch_ipex/csrc/utils.h"
#include "DevOPs.h"

using namespace at;

class NewLinearOp : public torch::autograd::Function<NewLinearOp> {
  public:
      static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        at::Tensor weight,
        at::Tensor bias) {
        ctx->save_for_backward({input, weight, bias});
        if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
          return torch_ipex::cpu::AtenIpexCPUDev::dil_linear(input, weight, bias);
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
 
      if (torch_ipex::check_auto_dnnl() && input.device().type() == c10::DeviceType::DPCPP) {
        grad_input = torch_ipex::cpu::AtenIpexCPUDev::dil_linear_backward_input(
            input.sizes(), grad_output, weight);
        std::tie(grad_weight, grad_bias) = torch_ipex::cpu::AtenIpexCPUDev::dil_linear_backward_weights(
            grad_output, input, weight, bias.defined());
      } else {
        auto grad_input = grad_output.mm(weight);
        auto grad_weight = grad_output.t().mm(input);
        if (bias.defined()) {
          grad_bias = grad_output.sum(0);
        }
      }
      return {grad_input, grad_weight, grad_bias};
    }
};
