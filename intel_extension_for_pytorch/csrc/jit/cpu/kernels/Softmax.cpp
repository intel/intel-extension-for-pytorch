#include "Softmax.h"
#include "csrc/aten/cpu/AddSoftmax.h"
#include "csrc/cpu/ideep/IDeepConversions.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/utils/ipex_op_profile.h"

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
  IPEX_RECORD_FUNCTION("dil_softmax", c10::ArrayRef<c10::IValue>({}));

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

// Dispatch inplace softmax to oneDNN path for jit inference
at::Tensor& dil_softmax_(
    at::Tensor& input,
    const int64_t dim,
    const at::IValue& dtype) {
  IPEX_RECORD_FUNCTION("dil_softmax_", c10::ArrayRef<c10::IValue>({}));

  auto half_to_float = false;
  if (!dtype.isNone()) {
    auto outtype = dtype.toScalarType();
    auto intype = input.scalar_type();
    AT_ASSERTM(
        intype != at::ScalarType::Half,
        "softmax with half to float conversion is not supported on Mkldnn");
    input = input.toType(outtype);
    softmax_impl_(input, dim);
    return input;
  }
  softmax_impl_(input, dim);
  return input;
}

} // namespace cpu
} // namespace torch_ipex
