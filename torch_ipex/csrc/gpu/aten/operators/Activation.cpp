#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>

#include <core/DPCPP.h>

#include "Eltwise.hpp"
#include "Loops.h"

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
static inline bool is_contiguous(const int64_t* strides) {
  return strides[0] == sizeof(scalar_t) && strides[1] == sizeof(scalar_t) &&
      strides[2] == sizeof(scalar_t);
}

template <typename scalar_t>
static void dpcpp_threshold_kernel(TensorIterator& iter) {
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    dpcpp_eltwise_backward<mkldnn::algorithm::eltwise_relu>(
        data[0], data[1], data[2], n, 0.0f, 0.0f);
  };
  iter.serial_for_each(loop, {0L, iter.numel()});
}

// Note: dpcpp compiler does not support uname type in template.
class SyclOpThreshold {};

static void threshold_kernel(
    TensorIterator& iter,
    Scalar threshold_scalar,
    Scalar value_scalar) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, iter.dtype(), "threshold", [&] {
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
          dpcpp_threshold_kernel<scalar_t>(iter);
        } else {
          dpcpp_kernel_for_tensor_iter<SyclOpThreshold>(
              iter, [=](scalar_t x, scalar_t other) -> scalar_t {
                return x <= threshold ? value : other;
              });
        }
      });
}

} // namespace impl

Tensor relu(const Tensor& self) {
  return at::threshold(self, 0, 0);
}

Tensor& relu_(Tensor& self) {
  return at::threshold_(self, 0, 0);
}

static Tensor threshold_out(
    optional<Tensor> opt_result,
    const Tensor& self,
    Scalar threshold,
    Scalar value,
    const Tensor& other) {
  Tensor result = opt_result.value_or(Tensor());
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::threshold_kernel(iter, threshold, value);
  return iter.output();
}

Tensor& threshold_(Tensor& self, Scalar threshold, Scalar value) {
  threshold_out(make_optional(self), self, threshold, value, self);
  return self;
}

Tensor threshold(const Tensor& self, Scalar threshold, Scalar value) {
  return threshold_out(nullopt, self, threshold, value, self);
}

Tensor threshold_out(
    Tensor& result,
    const Tensor& self,
    Scalar threshold,
    Scalar value) {
  threshold_out(make_optional(result), self, threshold, value, self);
  return result;
}

Tensor threshold_backward(
    const Tensor& grad,
    const Tensor& self,
    Scalar threshold) {
  return threshold_out(nullopt, self, threshold, 0, grad);
}

DPCPP_DEF_K1(DPCPPOpHardShrink);
Tensor hardshrink(const Tensor& self, Scalar lambd_) {
  auto out_tensor = at::empty_like(self);

  auto iter = TensorIterator();
  iter.add_output(out_tensor);
  iter.add_input(self);
  iter.build();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "hardshrink", [&] {
    auto lambd = lambd_.to<scalar_t>();
    dpcpp_kernel_for_tensor_iter<DPCPP_K(DPCPPOpHardShrink)>(
        iter, [=](scalar_t x) -> scalar_t {
          return (x >= -lambd && x <= lambd) ? scalar_t(0) : x;
        });
  });
  return out_tensor;
}

DPCPP_DEF_K1(DPCPPOpHardShrinkBackward);
Tensor hardshrink_backward(
    const Tensor& grad,
    const Tensor& self,
    Scalar lambd_) {
  auto out_tensor = at::empty_like(grad);

  auto iter = TensorIterator();
  iter.add_output(out_tensor);
  iter.add_input(grad);
  iter.add_input(self);
  iter.build();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "hardshrink_backward", [&] {
        auto lambd = lambd_.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(DPCPPOpHardShrinkBackward)>(
            iter, [=](scalar_t grad_output, scalar_t x) -> scalar_t {
              return (x >= -lambd && x <= lambd) ? scalar_t(0) : grad_output;
            });
      });
  return out_tensor;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
