#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include "comm/ATDispatch.h"
#include "runtime/Utils.h"
#include "utils/DPCPP.h"

namespace at {
namespace AtenIpexTypeNestedTensorXPU {

std::vector<at::Tensor> unbind_int(const at::Tensor& self, int64_t dim) {
  TORCH_CHECK(
      dim == 0,
      "NestedTensor can only be unbound along dimension 0 ",
      "got dimension ",
      dim,
      " instead.");

  auto self_ptr = native::get_nested_tensor_impl(self);
  int64_t ntensors = self_ptr->size(0);
  std::vector<at::Tensor> result_tensors(ntensors);
  if (ntensors == 0) {
    return result_tensors;
  }
  // This returns a differentiable view of self as a regular tensor
  auto buffer = self.values();
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
                           strides = NestedTensor_get_strides(self_ptr);
  int64_t* offsets_ptr = self_ptr->get_storage_offsets().data_ptr<int64_t>();
  for (const int64_t i : c10::irange(ntensors)) {
    result_tensors[i] = buffer.as_strided(sizes[i], strides[i], offsets_ptr[i]);
  }
  return result_tensors;
}

} // namespace AtenIpexTypeNestedTensorXPU
} // namespace at