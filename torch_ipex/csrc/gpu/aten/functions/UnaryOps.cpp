#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <utils/Numerics.h>
#include <utils/Pointwise.h>


using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {
} // namespace impl

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


#define DEFINE_IPEX_OUT_INPLACE_FLOAT_TYPES_OPS(op, func, real)               \
  DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(op##_out, func, real)                       \
                                                                              \
  Tensor & op##_(Tensor & self) {                                             \
    return at::op##_out(self, self);                                          \
  }


#define DEFINE_IPEX_ALL_TYPES_CALLABLE_1_OPS(op, callable)                    \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_CALLABLE_1(op, callable)                              \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & self, Scalar value) {                                  \
    AT_DISPATCH_ALL_TYPES(self.scalar_type(), #op,                            \
        [&]() {                                                               \
          impl::op<scalar_t>(self, self, value.to<scalar_t>());               \
        }                                                                     \
    );                                                                        \
    return self;                                                              \
  }

#define DEFINE_IPEX_FLOAT_TYPES_CALLABLE_1_OPS(op, callable)                  \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_CALLABLE_1(op, callable)                              \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & self, Scalar value) {              \
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), #op,                       \
        [&]() {                                                               \
          impl::op<scalar_t>(self, self, value.to<scalar_t>());               \
        }                                                                     \
    );                                                                        \
    return self;                                                              \
  }


#define DEFINE_IPEX_OUT_ALL_TYPES_CALLABLE_1_OPS(op, callable)                \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_CALLABLE_1(op, callable)                              \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & out, const Tensor & self, Scalar value) {              \
    AT_DISPATCH_ALL_TYPES(self.scalar_type(), #op,                            \
        [&]() {                                                               \
          impl::op<scalar_t>(out, self, value.to<scalar_t>());                \
        }                                                                     \
    );                                                                        \
    return out;                                                               \
  }

#define DEFINE_IPEX_OUT_FLOAT_TYPES_CALLABLE_1_OPS(op, callable)              \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_CALLABLE_1(op, callable)                              \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & out, const Tensor & self, Scalar value) {              \
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), #op,                       \
        [&]() {                                                               \
          impl::op<scalar_t>(out, self, value.to<scalar_t>());                \
        }                                                                     \
    );                                                                        \
    return out;                                                               \
  }


#define DEFINE_IPEX_ALL_TYPES_CALLABLE_2_OPS(op, callable)                    \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_CALLABLE_2(op, callable)                              \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & self, Scalar val1, Scalar val2) {                      \
    AT_DISPATCH_ALL_TYPES(self.scalar_type(), #op,                            \
        [&]() {                                                               \
          impl::op<scalar_t>(self, self, val1.to<scalar_t>(), val2.to<scalar_t>()); \
        }                                                                     \
    );                                                                        \
    return self;                                                              \
  }

#define DEFINE_IPEX_FLOAT_TYPES_CALLABLE_2_OPS(op, callable)                  \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_CALLABLE_2(op, callable)                              \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & self, Scalar val1, Scalar val2) {                      \
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), #op,                       \
        [&]() {                                                               \
          impl::op<scalar_t>(self, self, val1.to<scalar_t>(), val2.to<scalar_t>()); \
        }                                                                     \
    );                                                                        \
    return self;                                                              \
  }


#define DEFINE_IPEX_OUT_ALL_TYPES_CALLABLE_2_OPS(op, callable)                \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_CALLABLE_2(op, callable)                              \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & out, const Tensor & self, Scalar val1, Scalar val2) {  \
    AT_DISPATCH_ALL_TYPES(self.scalar_type(), #op,                            \
        [&]() {                                                               \
          impl::op<scalar_t>(out, self, val1.to<scalar_t>(), val2.to<scalar_t>()); \
        }                                                                     \
    );                                                                        \
    return out;                                                               \
  }

#define DEFINE_IPEX_OUT_FLOAT_TYPES_CALLABLE_2_OPS(op, callable)              \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_CALLABLE_2(op, callable)                              \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & out, const Tensor & self, Scalar val1, Scalar val2) {  \
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), #op,                       \
        [&]() {                                                               \
          impl::op<scalar_t>(out, self, val1.to<scalar_t>(), val2.to<scalar_t>());                \
        }                                                                     \
    );                                                                        \
    return out;                                                               \
  }

DEFINE_IPEX_OUT_ALL_TYPES_OPS(abs_out, Numerics<scalar_t>::abs, Real);
DEFINE_IPEX_OUT_ALL_TYPES_OPS(neg_out, Numerics<scalar_t>::neg, Real);

DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(cos_out, Numerics<scalar_t>::cos, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(sin_out, Numerics<scalar_t>::sin, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(cosh_out, Numerics<scalar_t>::cosh, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(sinh_out, Numerics<scalar_t>::sinh, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(acos_out, Numerics<scalar_t>::acos, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(asin_out, Numerics<scalar_t>::asin, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(floor_out, Numerics<scalar_t>::floor, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(expm1_out, Numerics<scalar_t>::expm1, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(trunc_out, Numerics<scalar_t>::trunc, Real);
DEFINE_IPEX_OUT_FLOAT_TYPES_OPS(round_out, Numerics<scalar_t>::round, Real);

DEFINE_IPEX_OUT_INPLACE_FLOAT_TYPES_OPS(tan, Numerics<scalar_t>::tan, Real);
DEFINE_IPEX_OUT_INPLACE_FLOAT_TYPES_OPS(tanh, Numerics<scalar_t>::tanh, Real);
DEFINE_IPEX_OUT_INPLACE_FLOAT_TYPES_OPS(atan, Numerics<scalar_t>::atan, Real);
DEFINE_IPEX_OUT_INPLACE_FLOAT_TYPES_OPS(erf, Numerics<scalar_t>::erf, Real);
DEFINE_IPEX_OUT_INPLACE_FLOAT_TYPES_OPS(erfc, Numerics<scalar_t>::erfc, Real);
DEFINE_IPEX_OUT_INPLACE_FLOAT_TYPES_OPS(exp, Numerics<scalar_t>::exp, Real);

DEFINE_IPEX_ALL_TYPES_CALLABLE_1_OPS(clamp_max_, TensorMinValueOp);
DEFINE_IPEX_OUT_ALL_TYPES_CALLABLE_1_OPS(clamp_max_out, TensorMinValueOp);
DEFINE_IPEX_ALL_TYPES_CALLABLE_1_OPS(clamp_min_, TensorMaxValueOp);
DEFINE_IPEX_OUT_ALL_TYPES_CALLABLE_1_OPS(clamp_min_out, TensorMaxValueOp);
DEFINE_IPEX_OUT_ALL_TYPES_CALLABLE_2_OPS(clamp_min_max, TensorClampOp);

Tensor & clamp_out(Tensor & result, const Tensor & self,
    optional<Scalar> min, optional<Scalar> max) {
  if (min && max) {
    at::AtenIpexTypeDPCPP::clamp_min_max(result, self, *min, *max);
  } else if (max) {
    at::AtenIpexTypeDPCPP::clamp_max_out(result, self, *max);
  } else if (min) {
    at::AtenIpexTypeDPCPP::clamp_min_out(result, self, *min);
  } else {
    TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor & clamp_(Tensor & self, optional<Scalar> min, optional<Scalar> max) {
  return at::AtenIpexTypeDPCPP::clamp_out(self, self, min, max);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
