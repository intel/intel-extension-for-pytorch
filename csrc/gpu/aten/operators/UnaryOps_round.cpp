#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>

#include "comm/AccumulateType.h"
#include "comm/LoopsMeta.h"
#include "comm/Numerics.h"
#include "comm/Pairwise.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/zmath.h"

#include "Loops.h"
#include "LoopsTemplates.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_ALL_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_ALL_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);

Tensor& round_out(const Tensor& self, Tensor& out) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_round>(
      TensorIterator::unary_op,
      out,
      self,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.dtype(),
            "round",
            [&]() {
              dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
                return std::nearbyintf(static_cast<float>(a));
              });
            });
      },
      0.0f,
      0.0f,
      /*Onednn round only support float type*/ self.scalar_type() ==
          at::ScalarType::Float);
}

void round_decimals_out(const Tensor& self, int64_t decimals, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.dtype(),
      "round_decimals",
      [&]() {
        bool neg_flag = false;
        scalar_t ten_pow_decimals;
        if (decimals < 0) {
          decimals = -decimals;
          neg_flag = true;
        }
        ten_pow_decimals = static_cast<scalar_t>(std::pow(10, decimals));
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t a) -> scalar_t {
          return neg_flag
              ? std::nearbyint(a / ten_pow_decimals) * ten_pow_decimals
              : std::nearbyint(a * ten_pow_decimals) / ten_pow_decimals;
        });
      });
}

Tensor& round_out(const Tensor& self, int64_t decimals, Tensor& out) {
  if (decimals != 0) {
    at::AtenIpexTypeXPU::round_decimals_out(self, decimals, out);
  } else {
    at::AtenIpexTypeXPU::round_out(self, out);
  }
  return out;
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(fmod_out, TensorFmodOp);

Tensor fmod(const Tensor& self, const Scalar& other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeXPU::fmod_out(out, self, other);
}

Tensor& fmod_(Tensor& self, const Scalar& other) {
  return at::AtenIpexTypeXPU::fmod_out(self, self, other);
}

} // namespace AtenIpexTypeXPU
} // namespace at
