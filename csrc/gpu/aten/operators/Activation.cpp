#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Resize.h>
#include "ATen/OpMathType.h"

#include <ATen/xpu/XPUGeneratorImpl.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include "comm/ApplyUtils.h"

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include <oneDNN/oneDNN.h>
#include "DistributionTemplates.h"
#include "Loops.h"
#include "LoopsTemplates.h"
#include "RandomEngine.h"
#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <typename scalar_t>
inline scalar_t relu_forward(scalar_t self) {
  if (at::_isnan(self)) {
    return self;
  }
  return self > 0 ? self : static_cast<scalar_t>(0);
}

template <typename scalar_t>
inline scalar_t gelu_erf_forward(scalar_t x) {
  using opmath_t = at::opmath_type<scalar_t>;
  constexpr opmath_t kAlpha = M_SQRT1_2;
  return static_cast<opmath_t>(x) * opmath_t(0.5) *
      (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
}

template <typename scalar_t>
inline scalar_t gelu_tanh_forward(scalar_t x) {
  using opmath_t = at::opmath_type<scalar_t>;
  constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
  constexpr opmath_t kKappa = 0.044715;
  auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) *
      static_cast<opmath_t>(x);
  auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
  return opmath_t(0.5) * static_cast<opmath_t>(x) *
      (opmath_t(1) + Numerics<opmath_t>::tanh(inner));
}

template <typename scalar_t>
inline scalar_t gelu_quick_forward(scalar_t x) {
  using opmath_t = at::opmath_type<scalar_t>;
  return (scalar_t)(((opmath_t)x) / (1.0f + expf(-1.702f * (opmath_t)x)));
}

template <typename scalar_t>
struct SiluOutKernelDpcppFunctor {
  scalar_t operator()(scalar_t x) const {
    using accscalar_t = at::opmath_type<scalar_t>;
    const accscalar_t one = 1.0f;
    return static_cast<accscalar_t>(x) /
        (one + Numerics<accscalar_t>::exp(-static_cast<accscalar_t>(x)));
  }
};

Tensor& silu_out_kernel(const Tensor& self, Tensor& result) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_swish>(
      TensorIterator::unary_op,
      result,
      self,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.dtype(),
            "_silu_out",
            [&]() {
              SiluOutKernelDpcppFunctor<scalar_t> f;
              dpcpp_kernel_for_tensor_iter(iter, f);
            });
      },
      /* alpha = */ 1.0f);
}

} // namespace impl

template <typename scalar_t>
struct ReluFunctor {
  scalar_t operator()(scalar_t self) const {
    return impl::relu_forward<scalar_t>(self);
  }
};

Tensor relu(const Tensor& self) {
  Tensor result;
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_relu>(
      TensorIterator::unary_op, result, self, [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "relu",
            [&]() {
              ReluFunctor<scalar_t> f;
              dpcpp_kernel_for_tensor_iter(iter, f);
            });
      });
}

Tensor& relu_(Tensor& self) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_relu>(
      TensorIterator::unary_op, self, self, [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "relu_",
            [&]() {
              ReluFunctor<scalar_t> f;
              dpcpp_kernel_for_tensor_iter(iter, f);
            });
      });
}

template <typename scalar_t>
struct HardshrinkOutFunctor {
  scalar_t operator()(scalar_t x) const {
    return (x >= -_lambd && x <= _lambd) ? scalar_t(0) : x;
  }

  HardshrinkOutFunctor(scalar_t _lambd_) : _lambd(_lambd_) {}

 private:
  scalar_t _lambd;
};

Tensor& hardshrink_out(
    const Tensor& self,
    const Scalar& lambd,
    Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "hardshrink",
      [&]() {
        auto _lambd = lambd.to<scalar_t>();
        HardshrinkOutFunctor<scalar_t> f(_lambd);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return result;
}

Tensor hardshrink(const Tensor& self, const Scalar& lambd) {
  Tensor result = at::empty_like(self);
  return hardshrink_out(self, lambd, result);
}

template <typename scalar_t>
struct HardshrinkBackwardOutFunctor {
  scalar_t operator()(scalar_t grad_output, scalar_t x) const {
    return (x >= -_lambd && x <= _lambd) ? scalar_t(0) : grad_output;
  }

  HardshrinkBackwardOutFunctor(scalar_t _lambd_) : _lambd(_lambd_) {}

 private:
  scalar_t _lambd;
};

Tensor& hardshrink_backward_out(
    const Tensor& grad,
    const Tensor& self,
    const Scalar& lambd,
    Tensor& grad_input) {
  auto iter = TensorIterator::binary_op(grad_input, grad, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "hardshrink_backward_out",
      [&]() {
        auto _lambd = lambd.to<scalar_t>();
        HardshrinkBackwardOutFunctor<scalar_t> f(_lambd);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return grad_input;
}

Tensor hardshrink_backward(
    const Tensor& grad,
    const Tensor& self,
    const Scalar& lambd) {
  auto result = at::empty_like(grad);
  return hardshrink_backward_out(grad, self, lambd, result);
}

template <typename scalar_t>
struct GeluTanhOutFunctor {
  scalar_t operator()(scalar_t self) const {
    return impl::gelu_tanh_forward<scalar_t>(self);
  }
};

template <typename scalar_t>
struct GeluQuickOutFunctor {
  scalar_t operator()(scalar_t self) const {
    return impl::gelu_quick_forward<scalar_t>(self);
  }
};
template <typename scalar_t>
struct GeluErfOutFunctor {
  scalar_t operator()(scalar_t self) const {
    return impl::gelu_erf_forward<scalar_t>(self);
  }
};

Tensor& gelu_out(
    const Tensor& self,
    c10::string_view approximate,
    Tensor& result) {
  auto _approximate = at::native::get_gelutype_enum(approximate);
  if (_approximate == at::native::GeluType::Tanh) {
    return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_gelu_tanh>(
        TensorIterator::unary_op, result, self, [=](TensorIteratorBase& iter) {
          IPEX_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::BFloat16,
              at::ScalarType::Half,
              iter.dtype(),
              "gelu",
              [&]() {
                GeluTanhOutFunctor<scalar_t> f;
                dpcpp_kernel_for_tensor_iter(iter, f);
              });
        });
  } else {
    return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_gelu_erf>(
        TensorIterator::unary_op, result, self, [=](TensorIteratorBase& iter) {
          IPEX_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::BFloat16,
              at::ScalarType::Half,
              iter.dtype(),
              "gelu",
              [&]() {
                GeluErfOutFunctor<scalar_t> f;
                dpcpp_kernel_for_tensor_iter(iter, f);
              });
        });
  }
}

Tensor gelu_quick_out(const Tensor& self, Tensor& result) {
  bool out_defined = result.defined();
  auto iter = TensorIterator::unary_op(result, self);
  auto kernel_fn = [=](TensorIteratorBase& iter) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "gelu_quick",
        [&]() {
          GeluQuickOutFunctor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  };
  kernel_fn(iter);
  if (!out_defined)
    return iter.output();
  return result;
}

Tensor gelu_quick(const Tensor& self) {
  Tensor result;
  return gelu_quick_out(self, result);
}

Tensor& silu_out(const Tensor& self, Tensor& output) {
  return impl::silu_out_kernel(self, output);
}

template <typename scalar_t>
struct MishBackwardFunctor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using accscalar_t = at::opmath_type<scalar_t>;
    const accscalar_t dy_acc = static_cast<accscalar_t>(dy);
    const accscalar_t x_acc = static_cast<accscalar_t>(x);
    const accscalar_t s_acc =
        accscalar_t(1) / (accscalar_t(1) + Numerics<accscalar_t>::exp(-x_acc));
    const accscalar_t t_acc = std::tanh(
        Numerics<accscalar_t>::log1p(Numerics<accscalar_t>::exp(x_acc)));
    return dy_acc * (t_acc + x_acc * s_acc * (accscalar_t(1) - t_acc * t_acc));
  }
};

Tensor mish_backward(const Tensor& grad_output, const Tensor& input) {
  Tensor grad_input = at::empty({0}, input.options());
  auto iter = TensorIterator::binary_op(grad_input, grad_output, input);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mish_backward_xpu",
      [&]() {
        MishBackwardFunctor<scalar_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return grad_input;
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "gelu_quick", AtenIpexTypeXPU::gelu_quick, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "gelu_quick_out", AtenIpexTypeXPU::gelu_quick_out, c10::DispatchKey::XPU);
}
} // namespace
} // namespace AtenIpexTypeXPU
} // namespace at
