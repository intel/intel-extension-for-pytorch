
#include <ATen/ATen.h>

#include <ATen/native/dpcpp/Loops.h>
#include <ATen/native/Activation.h>
#include <ATen/native/dpcpp/Eltwise.hpp>
namespace at { namespace native {

template <typename scalar_t>
static inline bool is_contiguous(const int64_t* strides)
{
  return strides[0] == sizeof(scalar_t) &&
         strides[1] == sizeof(scalar_t) &&
         strides[2] == sizeof(scalar_t);
}

template <typename scalar_t>
static void sycl_threshold_kernel(TensorIterator& iter) {
  auto loop = [&](char**data, const int64_t* strides, int64_t n) {
    sycl_eltwise_backward<mkldnn::algorithm::eltwise_relu>(
        data[0], data[1], data[2], n, 0.0f, 0.0f); };
  iter.serial_for_each(loop, {0L, iter.numel()});
}


//Note: sycl compiler does not support uname type in template.
class SyclOpThreshold{};

static void threshold_kernel(TensorIterator& iter, Scalar threshold_scalar, Scalar value_scalar) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "threshold", [&] {
    scalar_t threshold = threshold_scalar.to<scalar_t>();
    scalar_t value = value_scalar.to<scalar_t>();
    bool all_contiguous = true;
    //(TODO) temp relu solution for relu FP16 support
    bool use_fp16 = false;
    if (iter.dtype() == ScalarType::Half) {
      use_fp16 = true;
    }
    for (int i = 0; i < iter.ntensors(); i++) {
      all_contiguous = all_contiguous && iter.tensor(i).is_contiguous();
    }
    if (threshold == 0 && value == 0 && all_contiguous && !use_fp16
        /*is_contiguous<scalar_t>(iter.get_strides().data())*/) {
      sycl_threshold_kernel<scalar_t>(iter);
    } else {
      sycl_kernel_for_tensor_iter<SyclOpThreshold>(
        iter, [=](scalar_t x, scalar_t other) -> scalar_t {
          return x <= threshold ? value : other;
        });
    }
  });
}
REGISTER_DISPATCH(threshold_stub, &threshold_kernel);
}} //namepsace at::native
