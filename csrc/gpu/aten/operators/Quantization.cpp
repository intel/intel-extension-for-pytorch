#include <ATen/ATen.h>
#include <ATen/quantized/QTensorImpl.h>

#include <oneDNN/oneDNN.h>
#include <quantized/QTensor.h>
#include <quantized/Quantizer.h>
#include <tensor/Tensor.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename underlying_t>
struct _make_per_tensor_quantized_tensor_functor {
  scalar_t operator()(underlying_t value) const {
    return scalar_t(value);
  }
};

Tensor _make_per_tensor_quantized_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point);
  Tensor self_contig = self.contiguous();
  IPEX_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "make_per_tensor_quantized_tensor_dpcpp", [&]() {
        auto iter = TensorIteratorConfig()
                        .add_output(dst)
                        .add_input(self)
                        .check_all_same_dtype(false)
                        .build();
        _make_per_tensor_quantized_tensor_functor<scalar_t, underlying_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return dst;
}

template <typename scalar_t, typename underlying_t>
struct _make_per_channel_quantized_tensor_functor {
  scalar_t operator()(underlying_t value) const {
    return scalar_t(value);
  }
};

Tensor _make_per_channel_quantized_tensor(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  Tensor dst = at::_empty_per_channel_affine_quantized(
      self.sizes(),
      scales,
      zero_points,
      axis,
      self.options().dtype(toQIntType(self.scalar_type())));
  Tensor self_contig = self.contiguous();
  IPEX_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "make_per_channel_quantized_tensor_dpcpp", [&]() {
        auto iter = TensorIteratorConfig()
                        .add_output(dst)
                        .add_input(self)
                        .check_all_same_dtype(false)
                        .build();
        _make_per_channel_quantized_tensor_functor<scalar_t, underlying_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return dst;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

template <typename scalar_t, typename underlying_t>
struct int_repr_functor {
  underlying_t operator()(scalar_t value) const {
    return value.val_;
  }
};

Tensor int_repr(const Tensor& self) {
  Tensor dst;
  IPEX_DISPATCH_QINT_TYPES(self.scalar_type(), "int_repr_dpcpp", [&]() {
    dst = at::empty(
        self.sizes(),
        self.options().dtype(UNDERLYING_TYPE),
        self.suggest_memory_format());
    auto iter = TensorIteratorConfig()
                    .add_output(dst)
                    .add_input(self)
                    .check_all_same_dtype(false)
                    .build();
    int_repr_functor<scalar_t, underlying_t> f;
    AtenIpexTypeXPU::dpcpp_kernel_for_tensor_iter(iter, f);
  });
  return dst;
}

} // namespace AtenIpexTypeQuantizedXPU

} // namespace at
