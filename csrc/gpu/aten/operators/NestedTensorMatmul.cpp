#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

namespace at {
namespace AtenIpexTypeNestedTensorXPU {

Tensor bmm(const Tensor& self, const Tensor& mat2) {
  if (self.is_nested() && !mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a nested self and non-nested other");
  } else if (!self.is_nested() && mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a non-nested self and nested other");
  }
  // should have guaranteed that at least one is nested
  auto self_ptr = native::get_nested_tensor_impl(self);
  auto mat2_ptr = native::get_nested_tensor_impl(mat2);
  TORCH_CHECK(self_ptr->dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(mat2_ptr->dim() == 3, "batch2 must be a 3D tensor");
  int64_t ntensors = self_ptr->size(0), ntensors2 = mat2_ptr->size(0);
  TORCH_CHECK(
      ntensors == ntensors2,
      "Expected size for the 1st dimension of batch2 tensor to be: ",
      ntensors,
      " but got: ",
      ntensors2,
      ".");

  // create a contiguous output
  const Tensor& self_sizemat = self_ptr->get_nested_sizes();
  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();

  std::vector<IntArrayRef> self_sizes =
      at::native::NestedTensor_get_sizes(self_ptr);
  std::vector<IntArrayRef> mat2_sizes =
      at::native::NestedTensor_get_sizes(mat2_ptr);

  int64_t out_numel = 0;
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef &self_shape = self_sizes[i], &mat2_shape = mat2_sizes[i];
    const int64_t &self_size0 = self_shape[0], &self_size1 = self_shape[1],
                  &mat2_size0 = mat2_shape[0], &mat2_size1 = mat2_shape[1];
    TORCH_CHECK(
        self_size1 == mat2_size0,
        i,
        "-th nested matrices in batch cannot be multiplied (",
        self_size0,
        "x",
        self_size1,
        " and ",
        mat2_size0,
        "x",
        mat2_size1,
        ")");
    out_sizemat_ptr[0] = self_size0;
    out_sizemat_ptr[1] = mat2_size1;
    // out_sizemat_ptr is used for record the output size, since each
    // tensor inside nested tesnor dim is 2, so we need add 2 here.
    out_sizemat_ptr += 2;
    out_numel += self_size0 * mat2_size1;
  }
  const Tensor& self_buffer = self_ptr->get_unsafe_storage_as_tensor();
  const Tensor& mat2_buffer = mat2_ptr->get_unsafe_storage_as_tensor();
  Tensor out_buffer = self_buffer.new_empty(out_numel);
  Tensor output = native::wrap_buffer(out_buffer, out_sizemat);
  auto out_ptr = native::get_nested_tensor_impl(output);

  std::vector<IntArrayRef> self_strides = NestedTensor_get_strides(self_ptr);
  std::vector<IntArrayRef> mat2_strides = NestedTensor_get_strides(mat2_ptr);
  const int64_t* self_offsets_ptr =
      self_ptr->get_storage_offsets().data_ptr<int64_t>();
  const int64_t* mat2_offsets_ptr =
      mat2_ptr->get_storage_offsets().data_ptr<int64_t>();
  const int64_t* out_offsets_ptr =
      out_ptr->get_storage_offsets().data_ptr<int64_t>();

  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::mm_out(
        output_unbind[i],
        self_buffer.as_strided(
            self_sizes[i], self_strides[i], self_offsets_ptr[i]),
        mat2_buffer.as_strided(
            mat2_sizes[i], mat2_strides[i], mat2_offsets_ptr[i]));
  }
  return output;
}

} // namespace AtenIpexTypeNestedTensorXPU
} // namespace at