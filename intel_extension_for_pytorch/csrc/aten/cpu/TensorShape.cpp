#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/IListRef.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <csrc/utils/library.h>

#include "TensorAdvancedIndexing.h"
#include "TensorShape.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(cat_contig_stub);

inline void cat_check_no_zero_dim(
    const at::MaterializedITensorListRef& tensors) {
  size_t i = 0;
  for (const at::Tensor& t : tensors) {
    TORCH_CHECK(
        t.dim() > 0,
        "zero-dimensional tensor (at position ",
        i,
        ") cannot be concatenated");
    i++;
  }
}

inline c10::MemoryFormat cat_compute_output_memory_format(
    const at::MaterializedITensorListRef& inputs) {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (const at::Tensor& t : inputs) {
    auto f = t.suggest_memory_format();
    if (f == c10::MemoryFormat::Contiguous) {
      return f;
    }
    if (format.has_value() && format.value() != f) {
      return c10::MemoryFormat::Contiguous;
    }
    format = f;
  }
  return format.value();
}

at::Tensor& cat_out_cpu(
    at::TensorList tensors,
    int64_t dim,
    at::Tensor& result) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::cat_out_cpu\n");
#endif
  RECORD_FUNCTION("torch_ipex::cat_out_cpu", c10::ArrayRef<c10::IValue>({}));
  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible to cat empty tensors unless all the other tensors were
  // 1-dimensional, so we allowed these tensors to be "skipped".  We maintain
  // this behavior for backwards compatibility, but only for this specific size
  // (i.e. other empty sizes are not skipped).
  auto materialized = at::ITensorListRef(tensors).materialize();

  cat_check_no_zero_dim(materialized);
  dim = at::legacy_cat_wrap_dim(dim, tensors);

  // Checking names before the actual dimensions.
  auto maybe_outnames = at::namedinference::compute_cat_outnames(tensors);

  TORCH_CHECK(
      materialized.size() > 0,
      "torch.cat(): expected a non-empty list of Tensors");

  // Look for the first valid tensor.
  size_t valid = materialized.size();
  for (const auto i : c10::irange(materialized.size())) {
    if (!at::native::cat_should_skip_tensor(materialized[i].get())) {
      valid = i;
      break;
    }
  }

  bool all_contiguous = true;
  bool all_same_dtype = true;
  bool all_same_sizes_and_stride = true;
  auto memory_format = cat_compute_output_memory_format(materialized);

  // Compute what the output dtype should be:
  auto is_out_defined = result.defined();
  auto out_dtype = at::native::result_type(tensors);

  // If the output tensor is defined, we need to take it into account
  // when computing the actual output dtype and the flags.
  if (is_out_defined) {
    // Check for type promotion, if the output tensor is defined.
    TORCH_CHECK(
        canCast(out_dtype, result.scalar_type()),
        "torch.cat(): input types can't be cast to the desired output type ",
        result.scalar_type());
    out_dtype = result.scalar_type();
    all_contiguous = result.is_contiguous(memory_format);
  }

  // Fallback 'set_output' parameters.
  // (in case we don't find a valid tensor)
  DimVector sizes{0};
  at::TensorOptions options =
      materialized[0].get().options().dtype(out_dtype).memory_format(
          memory_format);

  // If we found a valid tensor, check whether the input tensors
  // are compatible, i.e. we can execute `cat` on them.
  bool found_valid_tensor = valid < materialized.size();
  if (found_valid_tensor) {
    TORCH_CHECK(
        dim <= materialized[valid].get().dim(),
        "torch.cat(): dimension ",
        dim,
        "out of range");

    // Compute the output tensor size.
    // It should have the same shape as any other valid tensor,
    // except in the dimension 'dim'.
    size_t size_at_dim = 0;
    for (const auto i : c10::irange(materialized.size())) {
      const at::Tensor& t = materialized[i];
      if (!at::native::cat_should_skip_tensor(t)) {
        at::native::check_cat_shape_except_dim(materialized[valid], t, dim, i);
        size_at_dim += t.size(dim);
        all_contiguous = all_contiguous && t.is_contiguous(memory_format);
        all_same_dtype = all_same_dtype && out_dtype == t.scalar_type();
        all_same_sizes_and_stride = all_same_sizes_and_stride &&
            t.sizes() == materialized[valid].get().sizes() &&
            t.strides() == materialized[valid].get().strides();
      } else {
        all_contiguous = false;
      }
    }

    // Actually set the output.
    sizes = materialized[valid].get().sizes().vec();
    sizes[dim] = size_at_dim;
    options =
        materialized[valid].get().options().dtype(out_dtype).memory_format(
            memory_format);
  }

  result = at::empty(sizes, options);
  // Checks for overlaps between the inputs and the output tensor.
  if (is_out_defined && found_valid_tensor) {
    at::assert_no_internal_overlap(result);
    for (const at::Tensor& t : materialized) {
      at::assert_no_overlap(result, t);
    }
  }

  if (result.numel() == 0) {
    return result;
  }

  // fast path when both inputs and result are contiguous and not empty
  ScalarType dtype = materialized[valid].get().scalar_type();
  bool serial_dtype =
      (dtype == ScalarType::Double || dtype == ScalarType::Float ||
       dtype == ScalarType::BFloat16);
  if (all_contiguous && all_same_dtype && serial_dtype) {
    cat_contig_stub(kCPU, result, materialized, dim, all_same_sizes_and_stride);
    return result;
  }

  int64_t offset = 0;
  if (all_same_sizes_and_stride && result.is_contiguous(memory_format) &&
      all_same_dtype) {
    const at::Tensor& source_slice = materialized[valid];
    auto slice_dim_size = source_slice.sizes()[dim];
    auto result_slice = result.narrow(dim, 0, slice_dim_size);
    auto result_slice_data = result_slice.data_ptr();
    auto result_stride_bytes =
        result.stride(dim) * elementSize(result.scalar_type());

    auto iter = at::TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .resize_outputs(false)
                    .add_output(result_slice)
                    .add_input(source_slice)
                    .enforce_safe_casting_to_output(true)
                    .build();

    for (const at::Tensor& tensor : materialized) {
      if (at::native::cat_should_skip_tensor(tensor)) {
        continue;
      }
      auto source_data = static_cast<char*>(tensor.data_ptr());
      auto result_data =
          static_cast<char*>(result_slice_data) + offset * result_stride_bytes;
      iter.unsafe_replace_operand(0, result_data);
      iter.unsafe_replace_operand(1, source_data);
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  } else {
    for (const at::Tensor& tensor : materialized) {
      if (at::native::cat_should_skip_tensor(tensor)) {
        continue;
      }
      auto slice_dim_size = tensor.sizes()[dim];
      auto result_slice = result.narrow(dim, offset, slice_dim_size);

      auto iter = at::TensorIteratorConfig()
                      .set_check_mem_overlap(false) // Already checked above
                      .resize_outputs(false)
                      .add_output(result_slice)
                      .add_input(tensor)
                      .promote_inputs_to_common_dtype(true)
                      .cast_common_dtype_to_outputs(true)
                      .enforce_safe_casting_to_output(true)
                      .build();
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  }

  return result;
}

at::Tensor cat_cpu(at::TensorList tensors, int64_t dim) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::cat_cpu\n");
#endif
  RECORD_FUNCTION("torch_ipex::cat_cpu", c10::ArrayRef<c10::IValue>({}));
  at::Tensor result;
  return cat_out_cpu(tensors, dim, result);
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::cat"), TORCH_FN((&torch_ipex::cpu::cat_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::cat.out"),
      TORCH_FN((&torch_ipex::cpu::cat_out_cpu)));
}

} // namespace cpu
} // namespace torch_ipex