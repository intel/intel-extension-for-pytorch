#include "Matmul.h"
#include "csrc/aten/cpu/Matmul.h"
#include "csrc/utils/utils.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

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
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_matmul_div_fallback", std::vector<c10::IValue>({}));
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
at::Tensor dil_matmul_div(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const c10::Scalar& div_input) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_matmul_div_scalar", std::vector<c10::IValue>({}));
#endif
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

} // namespace cpu
} // namespace torch_ipex
