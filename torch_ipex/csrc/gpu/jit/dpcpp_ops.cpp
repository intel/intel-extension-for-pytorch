#include <ATen/ATen.h>
#include <torch/csrc/autograd/record_function.h>
#include <dnnl.hpp>

#include <ATen/aten_ipex_type_dpcpp.h>
#include <ATen/ipex_type_dpcpp_customized.h>


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

at::Tensor mul_add(const at::Tensor& self,
    const at::Tensor& other, const at::Tensor& accumu, at::Scalar alpha) {
  return at::AtenIpexTypeDPCPP::mul_add(self, other, accumu, alpha);
}

at::Tensor q_conv2d_sum_relu(at::Tensor& accumu,
    const at::Tensor& input, const at::Tensor& packed_weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, double conv_scale, int64_t conv_zpoint, double sum_scale,
    int64_t sum_zpoint) {
  RECORD_FUNCTION("q_conv2d_sum_relu",
                  std::vector<c10::IValue>({input, packed_weight}));
  return at::AtenIpexTypeDPCPP::q_conv2d_sum_relu(accumu,
      input, packed_weight, stride, padding, dilation, groups, conv_scale,
      conv_zpoint, sum_scale, sum_zpoint);
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
