#include "Softmax.h"
#include "AddSoftmax.h"
#include "csrc/aten/cpu/Softmax.h"
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

// Dispatch softmax to oneDNN path for jit inference
at::Tensor dil_softmax(
    const at::Tensor& input,
    const int64_t dim,
    const at::IValue& dtype) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_softmax", std::vector<c10::IValue>({}));
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

} // namespace cpu
} // namespace torch_ipex
