#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <core/DPCPP.h>
#include <core/Context.h>
#include <utils/Numerics.h>

#include "Loops.h"


using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DPCPP_DEF_K1(softshrink_forward);
static void softshrink_forward(TensorIterator &iter, Scalar lambd) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "softshrink_forward", [&]() {
    auto lambd_data = lambd.to<scalar_t> ();
    dpcpp_kernel_for_tensor_iter<DPCPP_K(softshrink_forward)>(iter, [=](scalar_t x)-> scalar_t {
      if (x > lambd_data) {
        return (x - lambd_data);
      } else if (x < -lambd_data) {
        return (x + lambd_data);
      } else {
        return ScalarConvert<int, scalar_t>::to(0);
      }
    });
  });
}

DPCPP_DEF_K1(softshrink_backward);
static void softshrink_backward(TensorIterator &iter, Scalar lambd) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "softshrink_backward", [&]() {
    auto lambd_data = lambd.to<scalar_t> ();
    dpcpp_kernel_for_tensor_iter<DPCPP_K(softshrink_backward)>(iter, [=](scalar_t grad_output, scalar_t input)-> scalar_t {
      if (input > lambd_data || input < -lambd_data) {
        return grad_output;
      } else{
        return ScalarConvert<int, scalar_t>::to(0);
      }
    });
  });
}

} // namespace impl

Tensor & softshrink_out(Tensor & out, const Tensor & self, Scalar lambd) {
  checkBackend("softshrink_forward", {out}, self.type().backend());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(out);
  iter.add_input(self);
  iter.build();
  impl::softshrink_forward(iter, lambd);
  return out;
}

Tensor softshrink(const Tensor & self, Scalar lambd) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::softshrink_out(out, self, lambd);
}

Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  checkBackend("softshrink_backward", {grad_input, grad_output}, self.type().backend());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(grad_output);
  iter.add_input(self);
  iter.build();
  impl::softshrink_backward(iter, lambd);
  return grad_input;
}

Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return at::softshrink_backward_out(grad_input, grad_output, self, lambd);
}

}} // namespace at::AtenIpexTypeDPCPP
