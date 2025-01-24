#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <core/Allocator.h>
#include <core/Device.h>
#include <core/detail/ListUtils.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include "BitonicMergeSort.h"
#include "Loops.h"
#include "PSTLFunctions.h"
#include "ReduceOpStdVar.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace at::native;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

Tensor empty_dpcpp(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(
      options.backend() == at::Backend::XPU ||
      options.backend() == at::Backend::QuantizedXPU);
  // TORCH_INTERNAL_ASSERT(!options.is_variable()); // is_variable should have
  // been
  // "unpacked"

  auto* allocator = torch_ipex::xpu::dpcpp::getDeviceAllocator();
  int64_t nelements = torch_ipex::xpu::dpcpp::detail::prod_intlist(size);
  auto dtype = options.dtype();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);
  auto tensor = detail::make_tensor<TensorImpl>(
      storage_impl, options.computeDispatchKey(), dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  TORCH_CHECK(
      !(options.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");

  auto memory_format = options.memory_format_opt().value_or(
      optional_memory_format.value_or(MemoryFormat::Contiguous));

  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor empty_quantized(
    IntArrayRef size,
    const Tensor& qtensor,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  TensorOptions specified_options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  TORCH_CHECK(
      !(specified_options.has_memory_format() && memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");

  TensorOptions options = qtensor.options()
                              .merge_in(specified_options)
                              .merge_memory_format(memory_format);

  Tensor output;
  if (qtensor.qscheme() == kPerTensorAffine) {
    output = at::_empty_affine_quantized(
        size, options, qtensor.q_scale(), qtensor.q_zero_point());
  } else if (
      qtensor.qscheme() == kPerChannelAffine ||
      qtensor.qscheme() == kPerChannelAffineFloatQParams) {
    output = at::_empty_per_channel_affine_quantized(
        size,
        qtensor.q_per_channel_scales().to(options.device()),
        qtensor.q_per_channel_zero_points().to(options.device()),
        qtensor.q_per_channel_axis(),
        options);
  } else {
    TORCH_CHECK(
        false,
        "QScheme not supported by empty_quantized:",
        toString(qtensor.qscheme()));
  }
  return output;
}

Tensor empty_strided_dpcpp(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  check_size_nonnegative(size);
  auto t = empty_dpcpp({0}, options, c10::nullopt);
  resize_impl(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

} // namespace impl
Tensor empty(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  return AtenIpexTypeXPU::impl::empty_dpcpp(
      size, options, optional_memory_format);
}

Tensor empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return AtenIpexTypeXPU::impl::empty_strided_dpcpp(size, stride, options);
}

Tensor& std_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef _dim,
    const c10::optional<at::Scalar>& _correction,
    bool keepdim,
    Tensor& out) {
  return at::AtenIpexTypeXPU::std_var_out(
      "std", out, self, _dim, _correction, keepdim, true);
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor empty(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  return AtenIpexTypeXPU::impl::empty_dpcpp(
      size, options, optional_memory_format);
}

Tensor empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return AtenIpexTypeXPU::impl::empty_strided_dpcpp(size, stride, options);
}

Tensor empty(
    IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return empty(size, options, optional_memory_format);
}

Tensor empty_like(
    const Tensor& self,
    c10::optional<ScalarType> dtype = c10::nullopt,
    c10::optional<Layout> layout = c10::nullopt,
    c10::optional<Device> device = c10::nullopt,
    c10::optional<bool> pin_memory = c10::nullopt,
    c10::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt) {
  return at::native::empty_like_quantized(
      self, dtype, layout, device, pin_memory, optional_memory_format);
}

Tensor empty_quantized(
    IntArrayRef size,
    const Tensor& qtensor,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  return AtenIpexTypeXPU::impl::empty_quantized(
      size, qtensor, dtype, layout, device, pin_memory, memory_format);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
