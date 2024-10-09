#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <oneDNN/oneDNN.h>
#include <utils/CustomOperatorRegistration.h>
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

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

IPEX_OUT_ALL_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_ALL_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);

template <typename scalar_t>
static inline scalar_t nearbyint_wrapper(scalar_t a) {
  return static_cast<scalar_t>(::nearbyintf(static_cast<float>(a)));
}
static inline double nearbyint_wrapper(double a) {
  return ::nearbyint(a);
}

template <typename scalar_t>
struct round_out_functor {
  scalar_t operator()(scalar_t a) const {
    return nearbyint_wrapper(a);
  }
};

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
              round_out_functor<scalar_t> f;
              dpcpp_kernel_for_tensor_iter(iter, f);
            });
      },
      0.0f,
      0.0f,
      /*Onednn round only support float type*/ self.scalar_type() ==
          at::ScalarType::Float);
}

template <typename scalar_t>
struct round_decimals_out_functor {
  scalar_t operator()(scalar_t a) const {
    return neg_flag ? std::nearbyint(a / ten_pow_decimals) * ten_pow_decimals
                    : std::nearbyint(a * ten_pow_decimals) / ten_pow_decimals;
  }

  round_decimals_out_functor(bool neg_flag, scalar_t ten_pow_decimals)
      : neg_flag(neg_flag), ten_pow_decimals(ten_pow_decimals) {}

 private:
  bool neg_flag;
  scalar_t ten_pow_decimals;
};

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
        round_decimals_out_functor<scalar_t> f(neg_flag, ten_pow_decimals);
        dpcpp_kernel_for_tensor_iter(iter, f);
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

namespace {

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("ceil.out", TORCH_FN((&at::AtenIpexTypeXPU::ceil_out)));
  m.impl("floor.out", TORCH_FN((&at::AtenIpexTypeXPU::floor_out)));
}

} // namespace
