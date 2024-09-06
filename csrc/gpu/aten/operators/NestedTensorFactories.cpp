#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>

namespace at {
namespace AtenIpexTypeNestedTensorXPU {
Tensor alias(const Tensor& self) {
  auto* nt_impl = at::native::get_nested_tensor_impl(self);
  auto buffer = nt_impl->get_unsafe_storage_as_tensor();
  const auto& nested_sizes = nt_impl->get_nested_sizes();
  const auto& nested_strides = nt_impl->get_nested_strides();
  const auto& storage_offsets = nt_impl->get_storage_offsets();
  return at::detail::make_tensor<at::native::NestedTensorImpl>(
      c10::TensorImpl::VIEW,
      std::move(buffer),
      nested_sizes,
      nested_strides,
      storage_offsets);
}
} // namespace AtenIpexTypeNestedTensorXPU
} // namespace at
