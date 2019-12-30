#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>

#include <functions/Loops.h>
#include <core/SYCLApplyUtils.h>

namespace at { namespace native {

template <typename T>
struct TensorFillOp {
  TensorFillOp(T v) : val(v) {}
  inline void operator()(T& v) const { v = val; }

  const T val;
};

void fill_kernel_sycl(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.dtype(), "fill_sycl", [&] {
    at::sycl::SYCL_tensor_apply1<scalar_t>(
        iter.tensor(0), TensorFillOp<scalar_t>(value.to<scalar_t>()));
  });
}

Tensor& fill__sycl_out(Tensor& self, Scalar value) {
  auto iter = TensorIterator::nullary_op(self);
  fill_kernel_sycl(iter, value);
  return self;
}

Tensor& fill__sycl(Tensor& self, Scalar value) {
  return fill__sycl_out(self, value);
}

Tensor& fill__sycl(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return fill__sycl_out(self, value.item());
}

} // namespace native
} // namespace at
