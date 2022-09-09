#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>

#include "Mean.h"
#include "utils/library.h"

namespace torch_ipex {
namespace cpu {

at::Tensor mean_dim_impl(
    const at::Tensor& input,
    c10::OptionalIntArrayRef dim_opt,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  int64_t dim_prod = 1;
  if (dim_opt.has_value()) {
    auto dim = dim_opt.value();
    if (dim.size() == 0 || input.ndimension() == 0) {
      dim_prod = input.numel();
    } else {
      for (auto d : dim) {
        dim_prod *= input.size(d);
      }
    }
  }
  return at::sum(input, dim_opt, keepdim, dtype).div_(dim_prod);
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::mean.dim"),
      TORCH_FN((&torch_ipex::cpu::mean_dim_impl)));
}

} // namespace cpu
} // namespace torch_ipex
