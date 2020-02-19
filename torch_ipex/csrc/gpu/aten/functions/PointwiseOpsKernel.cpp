#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/PointwiseOps.h>

#include <core/SYCL.h>
#include <functions/Loops.h>


DP_DEF_K1(addcmul);
DP_DEF_K1(addcdiv);

namespace at { namespace native {

static void addcmul_sycl_kernel(TensorIterator &iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "addcmul_sycl", [&]() {
    auto alpha = value.to<scalar_t>();
    sycl_kernel_for_tensor_iter<DP_K(addcmul)>(iter, [alpha](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a + alpha * b * c;
    });
  });
}

static void addcdiv_sycl_kernel(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "addcdiv_sycl", [&]() {
    auto alpha = value.to<scalar_t>();
    sycl_kernel_for_tensor_iter<DP_K(addcdiv)>(iter, [alpha](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a + alpha * (b / c);
    });
  });
}

}} // namespace at::native

namespace at { namespace AtenIpexTypeDPCPP {
Tensor & addcmul_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value){
  //checkBackend("addcmul_cpu", out, self.options().backend());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(out);
  iter.add_input(self);
  iter.add_input(tensor1);
  iter.add_input(tensor2);
  iter.build();
  at::native::addcmul_sycl_kernel(iter, value);
  return out;
}
Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value){
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::addcmul_out(result, self, tensor1, tensor2, value);
}
Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value){
  return at::AtenIpexTypeDPCPP::addcmul_out(self, self, tensor1, tensor2, value);
}

Tensor & addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value){
  //checkBackend("addcdiv_cpu", out, self.options().backend());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(out);
  iter.add_input(self);
  iter.add_input(tensor1);
  iter.add_input(tensor2);
  iter.build();
  at::native::addcdiv_sycl_kernel(iter, value);
  return out;
}
Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value){
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::addcdiv_out(result, self, tensor1, tensor2, value);
}
Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value){
  return at::AtenIpexTypeDPCPP::addcdiv_out(self, self, tensor1, tensor2, value);
}
} // namespace AtenIpexTypeDPCPP
} // namespace at
