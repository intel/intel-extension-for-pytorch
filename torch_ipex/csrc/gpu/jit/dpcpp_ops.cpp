#include <ATen/ATen.h>
#include <torch/csrc/autograd/record_function.h>
#include <dnnl.hpp>

#include <ATen/aten_ipex_type_dpcpp.h>


namespace torch {
namespace jit {
namespace dpcpp {

at::Tensor& conv2d_sum(at::Tensor& accumu,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Scalar alpha) {
  RECORD_FUNCTION("conv2d_sum",
                  std::vector<c10::IValue>({input, weight, bias, accumu}));
  at::AtenIpexTypeDPCPP::convolution_sum(input, weight, bias,
      stride, padding, dilation, false, {{0, 0}}, groups, accumu, alpha);
  return accumu;
}

at::Tensor& conv2d_sum_relu(at::Tensor& accumu,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Scalar alpha) {
  RECORD_FUNCTION("conv2d_sum_relu",
                  std::vector<c10::IValue>({input, weight, bias, accumu}));
  at::AtenIpexTypeDPCPP::convolution_sum_relu(input, weight, bias,
      stride, padding, dilation, false, {{0, 0}}, groups, accumu, alpha);
  return accumu;
}

at::Tensor conv2d_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION("conv2d_relu",
                  std::vector<c10::IValue>({input, weight, bias}));
  return at::AtenIpexTypeDPCPP::convolution_relu(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean,
    const at::Tensor& running_var, bool train, double momentum, double eps, bool use_dnn) {
  return at::empty_like(input);
}

at::Tensor fold_weight(
    const at::Tensor& weight, const at::Tensor& bn_weight, const at::Tensor& running_var, float eps) {
  return at::empty_like(weight);
}

at::Tensor fold_bias(
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& bn_weight,
    const at::Tensor& bn_bias, const at::Tensor& running_mean, const at::Tensor& running_var, float eps) {
  return at::empty_like(bias);
}

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from, dnnl::memory::format_tag to, int64_t groups) {
  return at::empty_like(input);
}

} // dpcpp
} // jit
} // torch
