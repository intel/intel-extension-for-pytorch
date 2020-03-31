#pragma once

#include <ATen/Tensor.h>

#include "dil/dil.hpp"

namespace torch_ipex {
namespace cpu {

class AtenIpexCPUDev {
 public:
  static at::Tensor convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups);
  // aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor
  static at::Tensor mkldnn_convolution(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups);
  static std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask);
  static std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask);

  // For DNNL OPs
  static at::Tensor dil_convolution(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);
  static at::Tensor& dil_relu_(at::Tensor& input);
  static at::Tensor& dil_add_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& other, at::Scalar alpha);
  static at::Tensor dil_add(const at::Tensor& self, const at::Tensor& other, at::Scalar alpha);
  static at::Tensor & dil_add_(at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor& dil_mul_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& other);
  static at::Tensor dil_mul(const at::Tensor& self, const at::Tensor& other);
  static at::Tensor & dil_mul_(at::Tensor & self, const at::Tensor & other);
  static at::Tensor dil_bmm(const at::Tensor& self, const at::Tensor& mat2);
  static at::Tensor& dil_bmm_out(at::Tensor &result, const at::Tensor& batch1, const at::Tensor& batch2);
  static at::Tensor dil_mm(const at::Tensor& self, const at::Tensor& mat2);
  static at::Tensor& dil_mm_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& mat2);
  static at::Tensor dil_baddbmm(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor & batch2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor& baddbmm_common(at::Tensor &result, const dil::tensor &bias, const dil::tensor &x, const dil::tensor &w, at::Scalar beta, at::Scalar alpha);
  static at::Tensor& dil_baddbmm_out(at::Tensor &result, const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor dil_addmm(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor & batch2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor& dil_addmm_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor dil_addbmm(const at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor& dil_addbmm_out(at::Tensor& result, const at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2, at::Scalar beta, at::Scalar alpha);
  static at::Tensor dil_linear(const at::Tensor& self, const at::Tensor& weight, const at::Tensor& bias);
  static at::Tensor dil_linear_backward_input(at::IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight);
  static std::tuple<at::Tensor, at::Tensor> dil_linear_backward_weights(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, bool bias_defined);
  static std::tuple<at::Tensor, at::Tensor, at::Tensor> dil_linear_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight, std::array<bool,3> output_mask);
};

}  // namespace cpu
}  // namespace torch_ipex
