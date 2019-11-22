#include <ATen/Dispatch.h>
#include <ATen/native/dpcpp/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>

#include <ATen/dpcpp/SYCLApplyUtils.h>

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

REGISTER_DISPATCH(fill_stub, &fill_kernel_sycl);

} // namespace native
} // namespace at
