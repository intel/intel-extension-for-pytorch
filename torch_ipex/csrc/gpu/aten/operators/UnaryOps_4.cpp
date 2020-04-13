#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <utils/Numerics.h>
#include <utils/Pairwise.h>
#include <utils/Pointwise.h>

#include "Loops.h"

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {

IPEX_OUT_FLOAT_AND_HALF_CALLABLE_0_UNARY_OPS(erfinv_out, TensorErfinvOp);

Tensor& erfinv_(Tensor& self) {
  return at::AtenIpexTypeDPCPP::erfinv_out(self, self);
}

Tensor erfinv(const Tensor& self) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::erfinv_out(result, self);
}

IPEX_OUT_FLOAT_AND_HALF_CALLABLE_0_UNARY_OPS(digamma_out, TensorDigammaOp);

Tensor& digamma_(Tensor& self) {
  return at::AtenIpexTypeDPCPP::digamma_out(self, self);
}

Tensor digamma(const Tensor& self) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::digamma_out(result, self);
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(remainder_out, TensorRemainderOp);

Tensor remainder(const Tensor& self, Scalar other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::remainder_out(out, self, other);
}

Tensor& remainder_(Tensor& self, Scalar other) {
  return at::AtenIpexTypeDPCPP::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(fmod_out, TensorFmodOp);

Tensor fmod(const Tensor& self, Scalar other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, Scalar other) {
  return at::AtenIpexTypeDPCPP::fmod_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_UNARY_OPS(sign_out, TensorSignOp);

Tensor sign(const Tensor& self) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::sign_out(out, self);
}

Tensor& sign_(Tensor& self) {
  return at::AtenIpexTypeDPCPP::sign_out(self, self);
}

Tensor& real_out(Tensor& out, const Tensor& self) {
  // TODO: support complex type
  // AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "real", [&]() {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "real", [&]() {
    out.resize_as_(self);
    DPCPP_tensor_apply2<scalar_t, scalar_t>(
        out, self, TensorRealOp<scalar_t>());
  });

  return out;
}

Tensor& conj_out(Tensor& out, const Tensor& self) {
  // TODO: support complex type
  // AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "conj", [&]() {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "conj", [&]() {
    out.resize_as_(self);
    DPCPP_tensor_apply2<scalar_t, scalar_t>(
        out, self, TensorConjOp<scalar_t>());
  });

  return out;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
