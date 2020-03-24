#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(min_out, TensorMinOp)

Tensor min(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::min_out(out, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(max_out, TensorMaxOp)

Tensor max(const Tensor& self, const Tensor& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::max_out(out, self, other);
}

IPEX_OUT_INT_CALLABLE_0_BINARY_OPS(__and___out, TensorBitAndOp)

Tensor __and__(const Tensor& self, const Tensor& other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__and___out(result, self, other);
}

Tensor& __iand__(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeDPCPP::__and___out(self, self, other);
}

IPEX_OUT_INT_CALLABLE_0_BINARY_OPS(__or___out, TensorBitOrOp)

Tensor __or__(const Tensor& self, const Tensor& other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__or___out(result, self, other);
}

Tensor& __ior__(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeDPCPP::__or___out(self, self, other);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
