#include "Shuffle.h"

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

at::Tensor dil_shuffle(
    const at::Tensor& self,
    at::IntArrayRef view_shape,
    int64_t dim0,
    int64_t dim1) {
  IPEX_RECORD_FUNCTION("dil_shuffle", c10::ArrayRef<c10::IValue>({}));
  ideep::tensor _self = itensor_view_from_dense(self);
  auto group_dim = dim0 < dim1 ? dim0 : dim1;
  auto groups = view_shape[group_dim];
  auto output = at::empty_like(self);
  ideep::tensor _output = itensor_view_from_dense(output);
  ideep::channel_shuffle_forward::compute(_self, _output, groups, group_dim);
  return output;
}

} // namespace cpu
} // namespace torch_ipex
