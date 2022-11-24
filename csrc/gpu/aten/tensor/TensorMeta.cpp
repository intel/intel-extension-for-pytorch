#include <ATen/native/Resize.h>
#include <tensor/TensorMeta.h>

namespace at {
namespace AtenIpexTypeXPU {
Tensor empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory);
} // namespace AtenIpexTypeXPU
} // namespace at

namespace at {
namespace AtenIpexTypeXPU {

void resize_tensor(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  TORCH_CHECK(
      options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      options.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      options.device() == out.device(),
      "Expected out tensor to have device ",
      options.device(),
      ", but got ",
      out.device(),
      " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(
          *options.memory_format_opt());
    }
  }
}

c10::optional<Tensor> maybe_create_tensor(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  if (out.strides() != strides) {
    return empty_strided(
        sizes,
        strides,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  }
  return c10::nullopt;
}

c10::optional<Tensor> set_strided(
    Tensor& output,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  resize_tensor(output, sizes, strides, options);
  auto new_tensor = maybe_create_tensor(output, sizes, strides, options);
  return new_tensor;
}

c10::optional<Tensor> set_contiguous(
    Tensor& output,
    IntArrayRef sizes,
    TensorOptions options) {
  auto strides = c10::contiguous_strides(sizes);
  return set_strided(output, sizes, strides, options);
}

void set_contiguous_no_create(
    Tensor& output,
    IntArrayRef sizes,
    TensorOptions options) {
  auto strides = c10::contiguous_strides(sizes);
  resize_tensor(output, sizes, strides, options);
}
} // namespace AtenIpexTypeXPU
} // namespace at