#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <dnnl.hpp>


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
  at::AtenIpexTypeXPU::convolution_sum(input, weight, bias,
      stride, padding, dilation, false, {{0, 0}}, groups, accumu, alpha);
  return accumu;
}

at::Tensor& conv2d_sum_relu(at::Tensor& accumu,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Scalar alpha) {
  RECORD_FUNCTION("conv2d_sum_relu",
                  std::vector<c10::IValue>({input, weight, bias, accumu}));
  at::AtenIpexTypeXPU::convolution_sum_relu(input, weight, bias,
      stride, padding, dilation, false, {{0, 0}}, groups, accumu, alpha);
  return accumu;
}

at::Tensor conv2d_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION("conv2d_relu",
                  std::vector<c10::IValue>({input, weight, bias}));
  return at::AtenIpexTypeXPU::convolution_relu(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor conv2d_sigmoid(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION("conv2d_sigmoid",
                  std::vector<c10::IValue>({input, weight, bias}));
  return at::AtenIpexTypeXPU::convolution_sigmoid(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor mul_add(const at::Tensor& self,
    const at::Tensor& other, const at::Tensor& accumu, at::Scalar alpha) {
  RECORD_FUNCTION("mul_add",
                  std::vector<c10::IValue>({self, other, accumu}));
  return at::AtenIpexTypeXPU::mul_add(self, other, accumu, alpha);
}

at::Tensor q_conv2d_sum_relu(at::Tensor& accumu,
    const at::Tensor& input, const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double conv_scale, int64_t conv_zpoint, double sum_scale,
    int64_t sum_zpoint) {
  RECORD_FUNCTION("q_conv2d_sum_relu",
                  std::vector<c10::IValue>({input, packed_weight}));
  return at::AtenIpexTypeXPU::q_conv2d_sum_relu(accumu,
      input, packed_weight, conv_scale,
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
