#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/ipex_type_dpcpp_customized.h>

#include "comm/ATDispatch.h"
#include <core/Context.h>
#include <core/TensorImplUtils.h>
#include <core/DPCPP.h>

#include "Loops.h"


DPCPP_DEF_K1(intrepr);

using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

int64_t q_zero_point(const Tensor& self) {
  return at::native::q_zero_point_quant(self);
}

double q_scale(const Tensor& self) {
  return at::native::q_scale_quant(self);
}

QScheme qscheme(const Tensor& self) {
  return at::native::qscheme_quant(self);
}

Tensor q_per_channel_scales(const Tensor& self) {
  return at::native::q_per_channel_scales(self);
}

Tensor q_per_channel_zero_points(const Tensor& self) {
  return at::native::q_per_channel_zero_points(self);
}

int64_t q_per_channel_axis(const Tensor& self) {
  return at::native::q_per_channel_axis(self);
}

Tensor& set_quantizer_(Tensor& self, ConstQuantizerPtr quantizer) {
  return at::native::set_quantizer_(self, quantizer);
}

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
    dpcpp_kernel_for_tensor_iter<DPCPP_K(intrepr)>(
        iter, [=](scalar_t value) -> underlying_t { return value.val_; });
  });
  return dst;
}

} // namespace AtenIpexTypeQuantizedXPU


namespace AtenIpexTypeXPU {
Tensor new_qtensor(
  IntArrayRef sizes,
  const TensorOptions &options,
  QuantizerPtr quantizer) {
  auto memory_format =
    options.memory_format_opt().value_or(MemoryFormat::Contiguous);

  at::Allocator *allocator = xpu::dpcpp::getDPCPPDeviceAllocator();

  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  native::check_size_nonnegative(sizes);
  int64_t nelements = at::prod_intlist(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
    isQIntType(typeMetaToScalarType(dtype)),
    dtype, " is not supported in new_qtensor on xpu device.");
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage = c10::make_intrusive<StorageImpl>(
    StorageImpl::use_byte_size_t(),
    size_bytes,
    allocator->allocate(size_bytes),
    allocator,
    /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
    storage, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);

  at::get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  at::get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);

  return tensor;
}
}
} // namespace at
