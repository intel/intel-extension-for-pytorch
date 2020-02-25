#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <utils/Numerics.h>
#include <utils/Pointwise.h>


using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {

#define DEFINE_IPEX_OUT_ALL_TYPES_OPS(op, func, real)                         \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_FUNC(op, func, real)                                  \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & out, const Tensor & self) {                            \
    AT_DISPATCH_ALL_TYPES(self.scalar_type(), #op,                            \
        [&]() {                                                               \
          impl::op<scalar_t>(out, self);                                      \
        }                                                                     \
    );                                                                        \
    return out;                                                               \
  }

#define DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(op, func, real)                       \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_FUNC(op, func, real)                                  \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & out, const Tensor & self) {                            \
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), #op,                       \
        [&]() {                                                               \
          impl::op<scalar_t>(out, self);                                      \
        }                                                                     \
    );                                                                        \
    return out;                                                               \
  }

DEFINE_IPEX_OUT_ALL_TYPES_OPS(abs_out, Numerics<scalar_t>::abs, Real);
DEFINE_IPEX_OUT_ALL_TYPES_OPS(neg_out, Numerics<scalar_t>::neg, Real);

DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(acos_out, Numerics<scalar_t>::acos, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(asin_out, Numerics<scalar_t>::asin, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(floor_out, Numerics<scalar_t>::floor, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(expm1_out, Numerics<scalar_t>::expm1, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(trunc_out, Numerics<scalar_t>::trunc, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(round_out, Numerics<scalar_t>::round, Real);

} // namespace AtenIpexTypeDPCPP
} // namespace at
