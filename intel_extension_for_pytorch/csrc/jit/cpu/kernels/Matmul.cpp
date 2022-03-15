#include "Matmul.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/utils/ipex_op_profile.h"

namespace torch_ipex {
namespace cpu {

/**
 * bmm oneDNN kernel
 *
 * @param tensor1
 * @param tensor2
 * @param out Optinal output provided by user for matmul
 * @attr Attribute for matmul oneDNN primitive
 * @return output Tensor.
 */
at::Tensor bmm_impl(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const ideep::attr_t& attr,
    const std::vector<ideep::tensor>& postop_tensors,
    const float dst_coeff = 1.0f) {
  auto tensor1_ = tensor1.is_contiguous() ? tensor1 : tensor1.contiguous();
  auto tensor2_ = tensor2.is_contiguous() ? tensor2 : tensor2.contiguous();
  const int64_t dim = tensor1.dim();
  const ideep::tensor mkldnn_input = itensor_view_from_dense(tensor1_);
  const ideep::tensor mkldnn_tensor2 = itensor_view_from_dense(tensor2_);

  auto output = out;
  if (!out.defined()) {
    std::vector<int64_t> output_size(dim);
    for (auto i = 0; i < dim - 1; i++) {
      output_size[i] = tensor1.size(i);
    }
    output_size[dim - 1] = tensor2.size(dim - 1);
    output = at::empty(output_size, tensor1.options());
  }
  ideep::tensor mkldnn_output = itensor_view_from_dense(output);
  ideep::matmul_forward::compute(
      mkldnn_input,
      mkldnn_tensor2,
      mkldnn_output,
      dst_coeff,
      1.0,
      ideep::scale_t(),
      ideep::scale_t(),
      ideep::scale_t(),
      attr,
      postop_tensors);

  return output;
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
at::Tensor dil_matmul_div(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const at::Tensor& div_input) {
  IPEX_RECORD_FUNCTION("dil_matmul_div_fallback", std::vector<c10::IValue>({}));

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
at::Tensor dil_matmul_div(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const c10::Scalar& div_input) {
  IPEX_RECORD_FUNCTION("dil_matmul_div_scalar", std::vector<c10::IValue>({}));

  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();
  if (dim_tensor1 == dim_tensor2 && dim_tensor1 >= 3) {
    float scale = 1.0f / div_input.to<float>();
    return bmm_impl(tensor1, tensor2, out, ideep::attr_t(), {}, scale);
  } else {
    return dil_matmul_div(
        tensor1, tensor2, out, at::native::wrapped_scalar_tensor(div_input));
  }
}

at::Tensor dil_bmm_add(
    const at::Tensor& input,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const c10::Scalar& alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_bmm_add", std::vector<c10::IValue>({}));
#endif
  auto batch1_dim = batch1.dim();
  auto batch2_dim = batch2.dim();
  if (batch1_dim == batch2_dim && batch1_dim >= 3) {
    auto _input = input.is_contiguous() ? input : input.contiguous();
    ideep::tensor onednn_input = itensor_view_from_dense(_input);

    auto op_attr = ideep::attr_t::fuse_binary(
        dnnl::algorithm::binary_add, onednn_input.get_desc());
    return bmm_impl(
        batch1, batch2, at::Tensor(), op_attr, {onednn_input}, 1.0f);
  } else {
    return at::baddbmm(input, batch1, batch2);
  }
}

} // namespace cpu
} // namespace torch_ipex
