#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

namespace at {
namespace AtenIpexTypeNestedTensorXPU {

Tensor bmm(const Tensor& self, const Tensor& mat2) {
  // dispatcher should have guaranteed that at least one is nested
  auto self_ptr = self.is_nested() ? native::get_nested_tensor_impl(self)
                                   : self.unsafeGetTensorImpl();
  auto mat2_ptr = mat2.is_nested() ? native::get_nested_tensor_impl(mat2)
                                   : mat2.unsafeGetTensorImpl();
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
  const Tensor& self_sizemat = self.is_nested()
      ? native::get_nested_tensor_impl(self)->get_nested_sizes()
      : native::get_nested_tensor_impl(mat2)->get_nested_sizes();

  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();

  int64_t out_numel = 0;
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef &self_shape = native::get_size_for_index(self, i),
                      &mat2_shape = native::get_size_for_index(mat2, i);
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
    out_sizemat_ptr += 2;
    out_numel += self_size0 * mat2_size1;
  }

  const Tensor& self_buffer = self.is_nested()
      ? native::get_nested_tensor_impl(self)->get_unsafe_storage_as_tensor()
      : self;
  const Tensor& mat2_buffer = mat2.is_nested()
      ? native::get_nested_tensor_impl(mat2)->get_unsafe_storage_as_tensor()
      : mat2;

  Tensor out_buffer = self_buffer.new_empty(out_numel);
  Tensor output = native::wrap_buffer(out_buffer, out_sizemat);
  auto out_ptr = native::get_nested_tensor_impl(output);

  const int64_t* out_offsets_ptr =
      out_ptr->get_storage_offsets().const_data_ptr<int64_t>();

  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::mm_out(
        output_unbind[i],
        self_buffer.as_strided(
            native::get_size_for_index(self, i),
            native::get_stride_for_index(self, i),
            native::get_offset_for_index(self, i)),
        mat2_buffer.as_strided(
            native::get_size_for_index(mat2, i),
            native::get_stride_for_index(mat2, i),
            native::get_offset_for_index(mat2, i)));
  }
  return output;
}
} // namespace AtenIpexTypeNestedTensorXPU
} // namespace at