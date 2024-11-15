#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include "comm/AccumulateType.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/quantized/QTensorImpl.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "PSTLFunctions.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "comm/ATDispatch.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
struct AssertAsyncKernelFunctor1 {
  void operator()() const {
    SYCL_KERNEL_ASSERT(input[0] != 0);
  }
  AssertAsyncKernelFunctor1(scalar_t* input_) : input(input_) {}

 private:
  scalar_t* input;
};

template <typename scalar_t>
void _assert_async_kernel(scalar_t* input) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgf) {
    AssertAsyncKernelFunctor1<scalar_t> kfn(input);
    cgf.single_task<decltype(kfn)>(kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

struct AssertAsyncKernelFunctor2 {
  void operator()() const {
    SYCL_KERNEL_ASSERT(input[0] != c10::complex<float>(0, 0));
  }
  AssertAsyncKernelFunctor2(c10::complex<float>* input_) : input(input_) {}

 private:
  c10::complex<float>* input;
};

template <>
void _assert_async_kernel(c10::complex<float>* input) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgf) {
    AssertAsyncKernelFunctor2 kfn(input);
    cgf.single_task<decltype(kfn)>(kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

struct AssertAsyncKernelFunctor3 {
  void operator()() const {
    SYCL_KERNEL_ASSERT(input[0] != c10::complex<double>(0, 0));
  }
  AssertAsyncKernelFunctor3(c10::complex<double>* input_) : input(input_) {}

 private:
  c10::complex<double>* input;
};

template <>
void _assert_async_kernel(c10::complex<double>* input) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgf) {
    AssertAsyncKernelFunctor3 kfn(input);
    cgf.single_task<decltype(kfn)>(kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

void _assert_async(const Tensor& self) {
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(
      n < 2, "Boolean value of Tensor with more than one value is ambiguous");
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_assert_async",
      [&]() { _assert_async_kernel<scalar_t>(self.data_ptr<scalar_t>()); });
}

// TODO: Ignore assert msg for now
void _assert_async(const Tensor& self_tensor, c10::string_view assert_msg) {
  at::AtenIpexTypeXPU::_assert_async(self_tensor);
}

template <typename scalar_t>
struct isneginf_out_functor {
  bool operator()(scalar_t a) const {
    return a == Numerics<scalar_t>::lower_bound();
  }
};

Tensor& isneginf_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(!self.is_complex(), "isneginf does not support complex inputs.");
  TORCH_CHECK(
      out.dtype() == at::kBool,
      "isneginf does not support non-boolean outputs.");
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    out.fill_(false);
  } else {
    TensorIterator iter;
    iter.build_borrowing_unary_force_boolean_op(out, self);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.input_dtype(),
        "isneginf",
        [&]() {
          isneginf_out_functor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
  return out;
}

template <typename scalar_t>
struct isposinf_out_functor {
  bool operator()(scalar_t a) const {
    return a == Numerics<scalar_t>::upper_bound();
  }
};

Tensor& isposinf_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(!self.is_complex(), "isposinf does not support complex inputs.");
  TORCH_CHECK(
      out.dtype() == at::kBool,
      "isposinf does not support non-boolean outputs.");
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    out.fill_(false);
  } else {
    TensorIterator iter;
    iter.build_borrowing_unary_force_boolean_op(out, self);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.input_dtype(),
        "isposinf",
        [&]() {
          isposinf_out_functor<scalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
