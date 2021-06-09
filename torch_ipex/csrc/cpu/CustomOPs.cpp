#include "torch_ipex/csrc/cpu/CustomOPs.h"
#include "torch_ipex/csrc/utils.h"
#include "Conv.h"
#include "Linear.h"
#include "Pooling.h"
#include "Matmul.h"
#include "Softmax.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor AtenIpexJITDev::dil_convolution_base(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_base", std::vector<c10::IValue>({}));
#endif
  return convolution_impl(input, weight, bias, stride, padding, dilation, groups, ideep::attr_t());
}

at::Tensor AtenIpexJITDev::dil_convolution_swish(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_swish", std::vector<c10::IValue>({}));
#endif
  return convolution_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_swish());
}

at::Tensor AtenIpexJITDev::dil_convolution_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sigmoid", std::vector<c10::IValue>({}));
#endif
  return convolution_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_sigmoid());
}

/**
 * Dispatch at::matmul + at::div pattern to ipex for jit inference, but only one-element 
 * tensor and channel dim boadcast is enabled in oneDNN 2.2.0 now. So, for simplicity,this path is just 
 * a fallback path now.
 * output(out) = (tensor1 * tensor2).div(div_input)
 *  
 * @param tensor1 
 * @param tensor2 
 * @param out Optinal output provided by user for matmul 
 * @param div_input Input Tensor for div 
 * @return Value for the fusion pattern output. 
 */
at::Tensor  AtenIpexJITDev::dil_matmul_div(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const at::Tensor& div_input) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_matmul_div_fallback", std::vector<c10::IValue>({}));
#endif
  if (out.defined()) {
    at::matmul_out(out, tensor1, tensor2);
    return out.div(div_input);
  } 
  auto output = at::matmul(tensor1, tensor2);
  return output.div(div_input);
      
 
}

/**
 *Dispatch at::matmul + at::div pattern to ipex for jit inference, but only bmm with same shape for 
 *tensor1 and tensor2 and scalar input for div will be dispatched to oneDNN kernel. Otherwise will fallback.
 *For oneDNN kernel, scalar input will be used as the scale attribute for matmul primitive.
 *output(out) = (tensor1 * tensor2).div(div_input_scalar).
 *ToDo: matmul + div scalar for matmul with other shape  
 *
 *@param tensor1
 *@param tensor2
 *@param out Optinal output provided by user for matmul
 *@param div_input Input scalar for div
 *@return Value for the fusion pattern output.
 */
at::Tensor  AtenIpexJITDev::dil_matmul_div(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const c10::Scalar& div_input) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_matmul_div_scalar", std::vector<c10::IValue>({}));
#endif
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();
  if (dim_tensor1 == dim_tensor2 && dim_tensor1 >= 3) { 
    float scale = 1.0 / div_input.to<float>(); 
    return bmm_impl(tensor1, tensor2, out, ideep::attr_t(), scale);
  } else {
    return AtenIpexJITDev::dil_matmul_div(tensor1, tensor2, out, at::native::wrapped_scalar_tensor(div_input));
  }
}

at::Tensor AtenIpexJITDev::dil_convolution_clamp(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    float lower_bound,
    float upper_bound) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_clamp", std::vector<c10::IValue>({}));
#endif
  return convolution_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_clamp(lower_bound, upper_bound));
}

at::Tensor AtenIpexJITDev::dil_convolution_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_relu", std::vector<c10::IValue>({}));
#endif
  return convolution_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_relu());
}

at::Tensor AtenIpexJITDev::dil_convolution_elu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    float alpha,
    at::Scalar scale,
    at::Scalar input_scale) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_elu", std::vector<c10::IValue>({}));
#endif
  auto scale_value = scale.to<float>();
  auto input_scale_value = input_scale.to<float>();
  return convolution_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_elu(scale_value, alpha, input_scale_value));
}

at::Tensor& AtenIpexJITDev::dil_convolution_sum(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sum", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  convolution_inplace_impl(
    input,
    weight,
    bias,
    accumu,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_sum(scale));
  return accumu;
}

at::Tensor& AtenIpexJITDev::dil_convolution_sum_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sum_relu", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  convolution_inplace_impl(
    input,
    weight,
    bias,
    accumu,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::residual(scale));
  return accumu;
}

at::Tensor AtenIpexJITDev::dil_max_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_max_pool2d", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(std::all_of(dilation.cbegin(), dilation.cend(), [](int64_t i) { return 1 == i; }),
      "dil_max_pool2d does not support dilation case");
  return pooling_impl(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      ideep::algorithm::pooling_max);
}

at::Tensor AtenIpexJITDev::dil_linear(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_linear", std::vector<c10::IValue>({}));
#endif
  return linear_impl(self, weight, bias, ideep::attr_t());
}

at::Tensor AtenIpexJITDev::dil_linear_fuse_eltwise(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_linear_fuse_eltwise", std::vector<c10::IValue>({}));
#endif
  return linear_impl(self, weight, bias, attr);
}


/**
 *Dispatch Linear + Add fusion pattern to ipex oneDNN kernel for inference mode.
 *This feature might improve performance for cases like residual learning blocks
 *Pattern: accum = accum * alpha + Linear(self, weight, bias) 
 *
 *@param self Activatin input for Linear  
 *@param weight Weight for Linear
 *@param bias Bias for Linear
 *@param accum One input for add operation, another is the output of Linear
 *@param alpha Scale for accum when doing add operation. 
 *
 *@return Value for the fusion pattern output. 
 */
at::Tensor AtenIpexJITDev::dil_linear_add(
    const at::Tensor& self, 
    const at::Tensor& weight, 
    const at::Tensor& bias, 
    at::Tensor& accumu, 
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_linear_add", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  return linear_inplace_impl(self, weight, bias, accumu, ideep::attr_t::fuse_sum(scale));
}

//Dispatch softmax to oneDNN path for jit inference
at::Tensor AtenIpexJITDev::dil_softmax(
    const at::Tensor& input,
    const int64_t dim,
    const at::IValue& dtype) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_softmax", std::vector<c10::IValue>({}));
#endif
  auto half_to_float = false;
  if (!dtype.isNone()) {
    auto outtype = dtype.toScalarType();
    auto intype = input.scalar_type();
    AT_ASSERTM(
      intype != at::ScalarType::Half,
      "softmax with half to float conversion is not supported on Mkldnn");
    at::Tensor converted = input.toType(outtype);
    return softmax_impl(converted, dim);
  }

  return softmax_impl(input, dim);
}

}  // namespace cpu
}  // namespace torch_ipex
