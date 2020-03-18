#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

#include <core/DPCPP.h>
#include <utils/Pointwise.h>
#include <functions/Loops.h>


using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

//Note: dpcpp compiler does not support uname type in template.
class SyclOpAdd{};
class SyclOpMul{};
class SyclOpDiv{};

static void add_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "add", [&]() {
    auto alpha = alpha_scalar.to<scalar_t> ();
    dpcpp_kernel_for_tensor_iter<SyclOpAdd>(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a + alpha * b;
        });
  });
}

static void sub_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  return add_kernel_dpcpp(iter, -alpha_scalar);
}

static void mul_kernel_dpcpp(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "mul", [&]() {
    dpcpp_kernel_for_tensor_iter<SyclOpMul>(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a * b;
        });
  });
}

static void div_kernel_dpcpp(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div", [&] {
      dpcpp_kernel_for_tensor_iter<SyclOpDiv>(iter,
        [](scalar_t a, scalar_t b)-> scalar_t {
        return a / b;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "div", [&]() {
      dpcpp_kernel_for_tensor_iter<SyclOpDiv>(iter,
        [](scalar_t a, scalar_t b)-> scalar_t {
        return a / b;
      });
    });
  }
}

// alpha_check
static inline void alpha_check(const TensorIterator& iter, Scalar alpha) {
  AT_CHECK(! alpha.isBoolean() || iter.dtype() == ScalarType::Bool,
              "Boolean alpha only supported for Boolean results.");
  AT_CHECK(isFloatingType(iter.dtype()) || alpha.isIntegral(true),
              "For integral input tensors, argument alpha must not be a floating point number.");
}

// scalar to tensor
static Tensor wrapped_scalar_tensor(Scalar scalar) {
  auto tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

// Basic checking for all sub functions.
static inline void sub_check(const Tensor& self, const Tensor& other) {
  AT_CHECK(self.scalar_type() != kBool || other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with two bool tensors is not supported. "
              "Use the `^` or `logical_xor()` operator instead.");
  AT_CHECK(self.scalar_type() != kBool && other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

} // namespace impl

Tensor& add_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  impl::alpha_check(iter, alpha);
  impl::add_kernel_dpcpp(iter,alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::alpha_check(iter, alpha);
  impl::add_kernel_dpcpp(iter,alpha);
  return iter.output();
}

Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return at::AtenIpexTypeDPCPP::add_out(self, self, other, alpha);
}

Tensor add(const Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeDPCPP::add(self, impl::wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeDPCPP::add_(self, impl::wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  impl::sub_check(self, other);
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  impl::alpha_check(iter, alpha);
  impl::sub_kernel_dpcpp(iter,alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  impl::sub_check(self, other);
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::alpha_check(iter, alpha);
  impl::sub_kernel_dpcpp(iter,alpha);
  return iter.output();
}

Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  return at::AtenIpexTypeDPCPP::sub_out(self, self, other, alpha);
}

Tensor rsub(const Tensor& self, const Tensor& other, Scalar alpha) {
  return at::AtenIpexTypeDPCPP::sub(other, self, alpha);
}

Tensor sub(const Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeDPCPP::sub(self, impl::wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_(Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeDPCPP::sub_(self, impl::wrapped_scalar_tensor(other), alpha);
}

Tensor rsub(const Tensor& self, Scalar other, Scalar alpha) {
  return at::AtenIpexTypeDPCPP::rsub(self, impl::wrapped_scalar_tensor(other), alpha);
}

Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  impl::mul_kernel_dpcpp(iter);
  return result;
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::mul_kernel_dpcpp(iter);
  return iter.output();
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeDPCPP::mul_out(self, self, other);
}

Tensor mul(const Tensor& self, Scalar other) {
  return at::AtenIpexTypeDPCPP::mul(self, impl::wrapped_scalar_tensor(other));
}

Tensor& mul_(Tensor& self, Scalar other) {
  return at::AtenIpexTypeDPCPP::mul_(self, impl::wrapped_scalar_tensor(other));
}

Tensor& div_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  impl::div_kernel_dpcpp(iter);
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::div_kernel_dpcpp(iter);
  return iter.output();
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return at::AtenIpexTypeDPCPP::div_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(min_out, TensorMinOp)

Tensor min(const Tensor & self, const Tensor & other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::min_out(out, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(max_out, TensorMaxOp)

Tensor max(const Tensor & self, const Tensor & other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::max_out(out, self, other);
}

IPEX_OUT_INT_CALLABLE_0_BINARY_OPS(__and___out, TensorBitAndOp)

Tensor __and__(const Tensor & self, const Tensor & other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__and___out(result, self, other);
}

Tensor & __iand__(Tensor & self, const Tensor & other) {
  return at::AtenIpexTypeDPCPP::__and___out(self, self, other);
}

IPEX_OUT_INT_CALLABLE_0_BINARY_OPS(__or___out, TensorBitOrOp)

Tensor __or__(const Tensor & self, const Tensor & other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__or___out(result, self, other);
}

Tensor & __ior__(Tensor & self, const Tensor & other) {
  return at::AtenIpexTypeDPCPP::__or___out(self, self, other);
}

DP_DEF_K1(tanh_backward);
Tensor & tanh_backward_out(
    Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(grad_output);
  iter.add_input(output);
  iter.build();

  AT_DISPATCH_ALL_TYPES(iter.dtype(), "tanh_backward_out", [&]() {
    dpcpp_kernel_for_tensor_iter<DP_K(tanh_backward)>(
        iter, [](scalar_t output, scalar_t z) -> scalar_t {
      return output * (1. - z*z);
    });
  });

  return grad_input;
}

Tensor tanh_backward(const Tensor & grad_output, const Tensor & output) {
  auto grad_input = at::empty({0}, grad_output.options());
  return at::tanh_backward_out(grad_input, grad_output, output);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(remainder_out, TensorCRemainderOp)

Tensor remainder(const Tensor & self, const Tensor & other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::remainder_out(out, self, other);
}

Tensor & remainder_(Tensor & self, const Tensor & other) {
  return at::AtenIpexTypeDPCPP::remainder_out(self, self, other);
}

IPEX_OUT_ALL_CALLABLE_0_BINARY_OPS(fmod_out, TensorCFmodOp)

Tensor fmod(const Tensor & self, const Tensor & other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::fmod_out(out, self, other);
}

Tensor & fmod_(Tensor & self, const Tensor & other) {
  return at::AtenIpexTypeDPCPP::fmod_out(self, self, other);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
