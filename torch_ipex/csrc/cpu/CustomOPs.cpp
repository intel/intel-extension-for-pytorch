#include "torch_ipex/csrc/cpu/CustomOPs.h"
#include "Conv.h"
#include "ConvTranspose.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "Matmul.h"
#include "Pooling.h"
#include "Softmax.h"
#include "torch_ipex/csrc/jit/cpu/kernels/AddLayerNorm.h"
#include "torch_ipex/csrc/jit/cpu/kernels/AddSoftmax.hpp"
#include "torch_ipex/csrc/utils.h"

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

/**
 * Dispatch at::matmul + at::div pattern to ipex for jit inference, but only
 * one-element tensor and channel dim boadcast is enabled in oneDNN 2.2.0 now.
 * So, for simplicity,this path is just a fallback path now. output(out) =
 * (tensor1 * tensor2).div(div_input)
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
 *Dispatch at::matmul + at::div pattern to ipex for jit inference, but only bmm
 *with same shape for tensor1 and tensor2 and scalar input for div will be
 *dispatched to oneDNN kernel. Otherwise will fallback. For oneDNN kernel,
 *scalar input will be used as the scale attribute for matmul primitive.
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
    float scale = 1.0f / div_input.to<float>();
    return bmm_impl(tensor1, tensor2, out, ideep::attr_t(), {}, scale);
  } else {
    return AtenIpexJITDev::dil_matmul_div(tensor1, tensor2, out, at::native::wrapped_scalar_tensor(div_input));
  }
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

/**
 * We tried to fuse Div+Matmul+Add+Softmax as a signel operator. But
 * the oneDNN matmul performance with binary postop is poor, then we splited
 * the fusion into two parts - Div+Matmul and Add+Softmax. When the oneDNN
 * fixes the performance issue, we can directly leverage oneDNN's
 * implementation.
 **/
at::Tensor AtenIpexJITDev::dil_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& rel_kv,
    const at::Scalar& alpha,
    const at::Scalar& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "AtenIpexJITDev::dil_mha_scores_calc", std::vector<c10::IValue>({}));
#endif
  auto _dim_per_head = dim_per_head.to<float>();
  auto _alpha = alpha.to<float>();
  auto qk = at::Tensor();

  auto q_dim = q.dim();
  auto k_dim = k.dim();
  qk = at::matmul(q, k);

  // Only support last dimension
  bool is_last_dim = (softmax_dim == -1);
  // Only support the non-last-dimension broadcast
  bool not_last_dim_broadcast = (rel_kv.size(rel_kv.ndimension() - 1) != 1);
  // Only support >=2D
  bool not_one_dim = q_dim >= 2;
  // Only support 64byte aligned
  bool aligned_64_bytes = rel_kv.size(rel_kv.ndimension() - 1) % 16 == 0;
  // Only support contiguous tensor
  bool is_contiguous = rel_kv.is_contiguous() && qk.is_contiguous();
  if (is_last_dim && not_last_dim_broadcast && not_one_dim &&
      aligned_64_bytes && is_contiguous && dtype.isNone() && _alpha == 1.0f) {
    return jit::cpu::kernels::DivAddSoftmax(qk, rel_kv, _dim_per_head);
  } else {
    qk = at::div(qk, dim_per_head);
    qk = at::add(qk, rel_kv, _alpha);
    return dil_softmax(qk, softmax_dim, dtype);
  }
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

at::Tensor AtenIpexJITDev::dil_shuffle(const at::Tensor &self,
                                       at::IntArrayRef view_shape, int64_t dim0,
                                       int64_t dim1) {
  ideep::tensor _self = itensor_view_from_dense(self);
  auto group_dim = dim0 < dim1 ? dim0 : dim1;
  auto groups = view_shape[group_dim];
  auto output = at::empty_like(self);
  ideep::tensor _output = itensor_view_from_dense(output);
  ideep::channel_shuffle_forward::compute(_self, _output, groups, group_dim);
  return output;
}

at::Tensor AtenIpexJITDev::dil_add_layernorm(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps,
    bool cuda_enable) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "AtenIpexJITDev::dil_add_layernorm", std::vector<c10::IValue>({}));
#endif
  // no broadcast
  bool no_broadcast = true;
  for (auto i = 0; i < a.ndimension(); i++) {
    if (a.size(i) != b.size(i)) {
      no_broadcast = false;
      break;
    }
  }
  // Only support 64byte aligned
  bool aligned_64_bytes = a.size(a.ndimension() - 1) % 16 == 0 &&
      b.size(b.ndimension() - 1) % 16 == 0;
  // Only support contiguous tensor
  bool is_contiguous = a.is_contiguous() && b.is_contiguous();
  if (no_broadcast && aligned_64_bytes && is_contiguous && alpha == 1.0f) {
    return jit::cpu::kernels::AddLayerNorm(
        a, b, alpha, normalized_shape, weight_opt, bias_opt, eps);
  } else {
    auto add_res = at::add(a, b, alpha);
    return at::layer_norm(add_res, normalized_shape, weight_opt, bias_opt, eps);
  }
}

}  // namespace cpu
}  // namespace torch_ipex
