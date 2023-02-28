#include <ATen/ATen.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <core/Generator.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Math.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

#include <ATen/Context.h>

#include <utils/DPCPP.h>
#include "comm/Numerics.h"
#include "comm/Pointwise.h"
#include "comm/ScalarOps.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& i0_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "i0_out",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          accscalar_t x = static_cast<accscalar_t>(a);
          return (scalar_t)(calc_i0(x));
        });
      });
  return out;
}

Tensor& special_ndtri_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "special_ndtri_out",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return calc_ndtri(a); });
      });
  return out;
}

Tensor& special_i0e_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "i0e",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          accscalar_t x = static_cast<accscalar_t>(a);
          return (scalar_t)(calc_i0e(x));
        });
      });
  return out;
}

Tensor& special_i1_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "i1", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          accscalar_t x = static_cast<accscalar_t>(a);
          return (scalar_t)(calc_i1(x));
        });
      });
  return out;
}

Tensor& special_i1e_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "i1e", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          accscalar_t x = static_cast<accscalar_t>(a);
          return (scalar_t)(calc_i1e(x));
        });
      });
  return out;
}

Tensor& special_entr_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "entr",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
          if (at::_isnan(x)) {
            return x;
          } else if (x > 0) {
            return -x * Numerics<scalar_t>::log(x);
          } else if (x == 0) {
            return 0;
          }
          return Numerics<scalar_t>::lower_bound();
        });
      });
  return out;
}

Tensor& special_erfcx_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "erfcx", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return calc_erfcx(a); });
      });
  return out;
}

Tensor& xlogy_out(const Tensor& self, const Tensor& other, at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "xlogy",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t y) -> scalar_t {
          if (at::_isnan(y)) {
            return NAN;
          }
          if (x == 0) {
            return 0;
          }
          return x * Numerics<scalar_t>::log(y);
        });
      });
  return out;
}

Tensor& special_xlog1py_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "xlog1py",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t y) -> scalar_t {
          if (at::_isnan(y)) {
            return NAN;
          }
          if (x == 0) {
            return 0;
          }
          return x * Numerics<scalar_t>::log1p(y);
        });
      });
  return out;
}

Tensor& special_bessel_j0_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "bessel_j0", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return bessel_j0_forward(a); });
      });
  return out;
}

Tensor& special_bessel_j1_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "bessel_j1", [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          if (a < scalar_t(0.0f)) {
            return -bessel_j1_forward(-a);
          }
          return bessel_j1_forward(a);
        });
      });
  return out;
}

Tensor& special_bessel_y0_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "bessel_y0", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return bessel_y0_forward(a); });
      });
  return out;
}

Tensor& special_bessel_y1_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "bessel_y1", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return bessel_y1_forward(a); });
      });
  return out;
}

Tensor& special_chebyshev_polynomial_t_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "chebyshev_polynomial_t",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t x, scalar_t n) -> scalar_t {
              return chebyshev_polynomial_t_forward<scalar_t>(x, n);
            });
      });
  return out;
}

Tensor& special_chebyshev_polynomial_u_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "chebyshev_polynomial_u",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t x, scalar_t n) -> scalar_t {
              return chebyshev_polynomial_u_forward<scalar_t>(x, n);
            });
      });
  return out;
}

Tensor& special_chebyshev_polynomial_v_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "chebyshev_polynomial_v",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t x, scalar_t n) -> scalar_t {
              return chebyshev_polynomial_v_forward<scalar_t>(x, n);
            });
      });
  return out;
}

Tensor& special_chebyshev_polynomial_w_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "chebyshev_polynomial_w",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t x, scalar_t n) -> scalar_t {
              return chebyshev_polynomial_w_forward<scalar_t>(x, n);
            });
      });
  return out;
}

Tensor& special_zeta_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "zeta", [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t q) -> scalar_t {
          return zeta<scalar_t>(x, q);
        });
      });
  return out;
}

Tensor& special_spherical_bessel_j0_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "spherical_bessel_j0",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return spherical_bessel_j0_forward(a);
        });
      });
  return out;
}

Tensor& special_hermite_polynomial_he_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "hermite_polynomial_he",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t n) -> scalar_t {
          return hermite_polynomial_he_forward<scalar_t>(x, n);
        });
      });
  return out;
}

Tensor& special_hermite_polynomial_h_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "", [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t n) -> scalar_t {
          return hermite_polynomial_h_forward<scalar_t>(x, n);
        });
      });
  return out;
}

Tensor& special_laguerre_polynomial_l_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "laguerre_polynomial_l",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t n) -> scalar_t {
          return laguerre_polynomial_l_forward<scalar_t>(x, n);
        });
      });
  return out;
}

Tensor& special_legendre_polynomial_p_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "legendre_polynomial_p",
      [&]() {
        dpcpp_kernel_with_scalars(iter, [](scalar_t x, scalar_t n) -> scalar_t {
          return legendre_polynomial_p_forward<scalar_t>(x, n);
        });
      });
  return out;
}

Tensor& special_modified_bessel_i0_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "modified_bessel_i0",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return modified_bessel_i0_forward(a);
        });
      });
  return out;
}

Tensor& special_modified_bessel_i1_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "modified_bessel_i1",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return modified_bessel_i1_forward(a);
        });
      });
  return out;
}

Tensor& special_modified_bessel_k0_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "modified_bessel_k0",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return modified_bessel_k0_forward(a);
        });
      });
  return out;
}

Tensor& special_modified_bessel_k1_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "modified_bessel_k1",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return modified_bessel_k1_forward(a);
        });
      });
  return out;
}

Tensor& special_scaled_modified_bessel_k0_out(
    const Tensor& self,
    at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "scaled_modified_bessel_k0",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return scaled_modified_bessel_k0_forward(a);
        });
      });
  return out;
}

Tensor& special_scaled_modified_bessel_k1_out(
    const Tensor& self,
    at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "scaled_modified_bessel_k1",
      [&]() {
        dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> scalar_t {
          return scaled_modified_bessel_k1_forward(a);
        });
      });
  return out;
}

Tensor& special_log_ndtr_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.common_dtype(), "log_ndtr", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t a) -> scalar_t { return calc_log_ndtr(a); });
      });
  return out;
}

Tensor& special_shifted_chebyshev_polynomial_t_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES(
      iter.common_dtype(), "shifted_chebyshev_polynomial_t", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t x, scalar_t n) -> scalar_t {
              return shifted_chebyshev_polynomial_t_forward<scalar_t>(x, n);
            });
      });
  return out;
}

Tensor& special_shifted_chebyshev_polynomial_u_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES(
      iter.common_dtype(), "shifted_chebyshev_polynomial_u", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t x, scalar_t n) -> scalar_t {
              return shifted_chebyshev_polynomial_u_forward<scalar_t>(x, n);
            });
      });
  return out;
}

Tensor& special_shifted_chebyshev_polynomial_v_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES(
      iter.common_dtype(), "shifted_chebyshev_polynomial_v", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t x, scalar_t n) -> scalar_t {
              return shifted_chebyshev_polynomial_v_forward<scalar_t>(x, n);
            });
      });
  return out;
}

Tensor& special_shifted_chebyshev_polynomial_w_out(
    const Tensor& self,
    const Tensor& other,
    at::Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  IPEX_DISPATCH_FLOATING_TYPES(
      iter.common_dtype(), "shifted_chebyshev_polynomial_w", [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t x, scalar_t n) -> scalar_t {
              return shifted_chebyshev_polynomial_w_forward<scalar_t>(x, n);
            });
      });
  return out;
}

Tensor& special_airy_ai_out(const Tensor& self, at::Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  IPEX_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "airy_ai", [&]() {
    dpcpp_kernel_for_tensor_iter(
        iter, [](scalar_t x) -> scalar_t { return airy_ai_forward(x); });
  });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
