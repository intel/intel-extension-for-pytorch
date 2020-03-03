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
    IMPLEMENT_POINTWISE_1_FUNC(op, func, real)                                  \
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
    IMPLEMENT_POINTWISE_1_FUNC(op, func, real)                                  \
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
    IMPLEMENT_POINTWISE_1_CALLABLE_1(op, callable)                              \
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
    IMPLEMENT_POINTWISE_1_CALLABLE_1(op, callable)                              \
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
    IMPLEMENT_POINTWISE_1_CALLABLE_1(op, callable)                              \
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
    IMPLEMENT_POINTWISE_1_CALLABLE_1(op, callable)                              \
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
    IMPLEMENT_POINTWISE_1_CALLABLE_2(op, callable)                              \
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
    IMPLEMENT_POINTWISE_1_CALLABLE_2(op, callable)                              \
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
    IMPLEMENT_POINTWISE_1_CALLABLE_2(op, callable)                              \
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
    IMPLEMENT_POINTWISE_1_CALLABLE_2(op, callable)                              \
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

IPEX_OUT_ALL_UNARY_FUNC_OPS(abs_out, Numerics<scalar_t>::abs, Real);
IPEX_OUT_ALL_UNARY_FUNC_OPS(neg_out, Numerics<scalar_t>::neg, Real);

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(cos_out, Numerics<scalar_t>::cos, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sin_out, Numerics<scalar_t>::sin, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(cosh_out, Numerics<scalar_t>::cosh, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sinh_out, Numerics<scalar_t>::sinh, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(acos_out, Numerics<scalar_t>::acos, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(asin_out, Numerics<scalar_t>::asin, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(expm1_out, Numerics<scalar_t>::expm1, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(trunc_out, Numerics<scalar_t>::trunc, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(round_out, Numerics<scalar_t>::round, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log_out, Numerics<scalar_t>::log, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log10_out, Numerics<scalar_t>::log10, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log1p_out, Numerics<scalar_t>::log1p, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log2_out, Numerics<scalar_t>::log2, Real);

IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(tan, Numerics<scalar_t>::tan, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(tanh, Numerics<scalar_t>::tanh, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(atan, Numerics<scalar_t>::atan, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(erf, Numerics<scalar_t>::erf, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(erfc, Numerics<scalar_t>::erfc, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(exp, Numerics<scalar_t>::exp, Real);

IPEX_ALL_CALLABLE_1_UNARY_OPS(clamp_max_, TensorMinValueOp);
IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(clamp_max_out, TensorMinValueOp);
IPEX_ALL_CALLABLE_1_UNARY_OPS(clamp_min_, TensorMaxValueOp);
IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(clamp_min_out, TensorMaxValueOp);
IPEX_OUT_ALL_CALLABLE_2_UNARY_OPS(clamp_min_max, TensorClampOp);

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
