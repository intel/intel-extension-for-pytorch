#include "Cumsum.h"
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <immintrin.h>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(cumsum_kernel_stub);

at::Tensor cumsum(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  auto casted_self = at::native::integer_upcast(self, dtype);
  at::Tensor result = at::empty_like(casted_self, at::MemoryFormat::Contiguous);

  // pointer to cumsum_kernel_impl(result, casted_self, dim, dtype);
  return cumsum_kernel_stub(kCPU, result, casted_self, dim, dtype);
}

at::Tensor& cumsum_(
    at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  // pointer to cumsum_kernel_impl(self, self, dim, dtype);
  cumsum_kernel_stub(kCPU, self, self, dim, dtype);

  return self;
}

at::Tensor& cumsum_out(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  // cumsum_kernel_impl(result, self.toType(result.scalar_type()), dim, dtype);
  cumsum_kernel_stub(
      kCPU, result, self.toType(result.scalar_type()), dim, dtype);

  return result;
}

} // namespace cpu

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor");
  m.impl("cumsum", c10::DispatchKey::CPU, torch_ipex::cpu::cumsum);
  m.def(
      "cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> "
      "Tensor(a!)");
  m.impl("cumsum_", c10::DispatchKey::CPU, torch_ipex::cpu::cumsum_);
  m.def(
      "cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, "
      "Tensor(a!) out) -> Tensor(a!)");
  m.impl("cumsum.out", c10::DispatchKey::CPU, torch_ipex::cpu::cumsum_out);
}

} // namespace
} // namespace torch_ipex
