#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>

#include "Mean.h"
#include "torch_ipex/csrc/library.h"

namespace torch_ipex {
namespace cpu {

at::Tensor mean_dim_impl(
    const at::Tensor& input,
    c10::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  auto input_ = input.is_contiguous() ? input : input.contiguous();
  int64_t dim_prod = 1;
  if (dim.size() == 0 || input_.ndimension() == 0) {
    dim_prod = input_.numel();
  } else {
    for (auto d : dim) {
      dim_prod *= input_.size(d);
    }
  }
  return at::sum(input_, dim, keepdim, dtype).div_(dim_prod);
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::mean.dim"),
      TORCH_FN((&torch_ipex::cpu::mean_dim_impl)));
}

} // namespace cpu
} // namespace torch_ipex
