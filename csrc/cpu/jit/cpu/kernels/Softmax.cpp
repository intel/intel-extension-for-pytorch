#include "Softmax.h"
#include "aten/AddSoftmax.h"
#include "ideep/IDeepConversions.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include <ideep.hpp>
#include "utils/onednn_utils.h"

namespace torch_ipex {
namespace cpu {

// softmax kernel for inference mode with oneDNN implementation
at::Tensor softmax_impl(const at::Tensor& input, const int64_t dim) {
  // do not support non-contiguous input, which should go into aten::softmax
  TORCH_CHECK(
      input.is_contiguous(),
      "ipex::softmax: Expected contiguous tensor input!");
  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, input.dim());
  ideep::tensor mkldnn_input = itensor_view_from_dense(input);
  auto output = at::empty_like(input);
  ideep::tensor mkldnn_output = itensor_view_from_dense(output);
  ideep::softmax_forward::compute(mkldnn_input, mkldnn_output, wrapped_dim);
  return output;
}

// inplace softmax kernel for inference mode with oneDNN implementation
void softmax_impl_(at::Tensor& input, const int64_t dim) {
  // do not support non-contiguous input, which should go into aten::softmax
  TORCH_CHECK(
      input.is_contiguous(),
      "ipex::softmax_: Expected contiguous tensor input!");
  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, input.dim());
  ideep::tensor mkldnn_input = itensor_view_from_dense(input);
  ideep::tensor mkldnn_output = itensor_view_from_dense(input);
  ideep::softmax_forward::compute(mkldnn_input, mkldnn_output, wrapped_dim);
}

// Dispatch softmax to oneDNN path for jit inference
at::Tensor dil_softmax(
    const at::Tensor& input,
    const int64_t dim,
    const at::IValue& dtype) {
  RECORD_FUNCTION("dil_softmax", c10::ArrayRef<c10::IValue>({}));

  if (!dtype.isNone()) {
    auto outtype = dtype.toScalarType();
    at::Tensor converted = input.toType(outtype);
    if (converted.scalar_type() == at::ScalarType::Half and
        !torch_ipex::utils::onednn_has_fp16_type_support()) {
      return at::softmax(converted, dim);
    }
    return softmax_impl(converted, dim);
  }
  if (input.scalar_type() == at::ScalarType::Half and
      !torch_ipex::utils::onednn_has_fp16_type_support()) {
    return at::softmax(input, dim);
  }
  return softmax_impl(input, dim);
}

// Dispatch inplace softmax to oneDNN path for jit inference
at::Tensor& dil_softmax_(
    at::Tensor& input,
    const int64_t dim,
    const at::IValue& dtype) {
  RECORD_FUNCTION("dil_softmax_", c10::ArrayRef<c10::IValue>({}));

  if (!dtype.isNone()) {
    auto outtype = dtype.toScalarType();
    input = input.toType(outtype);
    if (input.scalar_type() == at::ScalarType::Half and
        !torch_ipex::utils::onednn_has_fp16_type_support()) {
      at::softmax_out(input, input, dim);
      return input;
    }
    softmax_impl_(input, dim);
    return input;
  }
  if (input.scalar_type() == at::ScalarType::Half and
      !torch_ipex::utils::onednn_has_fp16_type_support()) {
    at::softmax_out(input, input, dim);
    return input;
  }
  softmax_impl_(input, dim);
  return input;
}

} // namespace cpu
} // namespace torch_ipex
