#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/record_function.h>

#include <c10/util/Unroll.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include <csrc/utils/library.h>

#include "TensorAdvancedIndexing.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(index_select_contig_stub);
DEFINE_DISPATCH(copy_stub);

at::Tensor& index_select_out_cpu_(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Tensor& result) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  auto numel = index.numel();
  TORCH_CHECK_INDEX(
      index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "index_select(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "index_select(): self and result must have the same scalar type");
  TORCH_CHECK(
      dim == 0 || dim < self.dim(),
      "index_select(): Indexing dim ",
      dim,
      " is out of bounds of tensor");
  at::assert_no_internal_overlap(result);
  at::assert_no_overlap(result, self);
  at::assert_no_overlap(result, index);
  auto result_size = self.sizes().vec();
  if (self.dim() > 0) {
    result_size[dim] = numel;
  }
  at::native::resize_output(result, result_size);

  auto index_contig = index.contiguous();

  if (self.dim() > 1) {
    if (numel == 0 || self.numel() == 0) {
      return result;
    }

    const auto st = self.scalar_type();
    if (result.is_contiguous() &&
        (st == at::kFloat || st == at::kDouble || st == at::kBFloat16)) {
      auto self_contig = self.contiguous();
      index_select_contig_stub(kCPU, result, self_contig, dim, index_contig);
      return result;
    }

    auto selfSlice = self.select(dim, 0);
    auto resultSlice = result.select(dim, 0);
    auto selfSlice_data = selfSlice.data_ptr();
    auto resultSlice_data = resultSlice.data_ptr();
    auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
    auto result_stride_bytes =
        result.stride(dim) * elementSize(result.scalar_type());
    auto self_dim_size = self.size(dim);
    auto slice_size = selfSlice.numel();

    auto iter = at::TensorIteratorConfig()
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(resultSlice)
                    .add_input(selfSlice)
                    .build();

    auto grain_size = at::internal::GRAIN_SIZE;
    auto outer_loop =
        // explicitly capture all required variables to work around windows
        // build
        // TODO: fix this when windows can correctly capture variables in nested
        // lambda
        [&index_contig,
         &iter,
         &self_dim_size,
         &selfSlice_data,
         &self_stride_bytes,
         &resultSlice_data,
         &result_stride_bytes](int64_t start, int64_t end) {
          auto sub_iter = at::TensorIterator(iter);
          AT_DISPATCH_INDEX_TYPES(
              index_contig.scalar_type(),
              "index_select_out_cpu_",
              [&index_contig,
               &start,
               &end,
               &sub_iter,
               &self_dim_size,
               &selfSlice_data,
               &self_stride_bytes,
               &resultSlice_data,
               &result_stride_bytes]() {
                auto index_data = index_contig.data_ptr<index_t>();
                for (const auto i : c10::irange(start, end)) {
                  auto self_i = index_data[i];
                  TORCH_CHECK_INDEX(
                      (self_i >= 0) && (self_i < self_dim_size),
                      "index out of range in self");
                  auto self_data = static_cast<char*>(selfSlice_data) +
                      self_i * self_stride_bytes;
                  auto result_data = static_cast<char*>(resultSlice_data) +
                      i * result_stride_bytes;
                  sub_iter.unsafe_replace_operand(0, result_data);
                  sub_iter.unsafe_replace_operand(1, self_data);
                  copy_stub(sub_iter.device_type(), sub_iter, false);
                };
              });
        };

    // parallel on inner loop in case the slice is large enough;
    // otherwise parallel on outer loop
    if (slice_size >= grain_size) {
      outer_loop(0, numel);
    } else {
      // use a fast loop when self and result are contiguous and of the same
      // data type
      if (iter.is_contiguous() && self.scalar_type() == result.scalar_type()) {
        auto slice_size_bytes = slice_size * elementSize(self.scalar_type());
        // explicitly capture all required variables to work around windows
        // build
        // TODO: fix this when windows can correctly capture variables in nested
        // lambda
        at::parallel_for(
            0,
            numel,
            grain_size / slice_size,
            [&index_contig,
             &slice_size_bytes,
             &self_dim_size,
             &selfSlice_data,
             &self_stride_bytes,
             &resultSlice_data,
             &result_stride_bytes](int64_t start, int64_t end) {
              AT_DISPATCH_INDEX_TYPES(
                  index_contig.scalar_type(),
                  "index_select_out_cpu_",
                  [&index_contig,
                   &slice_size_bytes,
                   &self_dim_size,
                   &selfSlice_data,
                   &self_stride_bytes,
                   &resultSlice_data,
                   &result_stride_bytes,
                   &start,
                   &end]() {
                    auto index_data = index_contig.data_ptr<index_t>();
                    for (const auto i : c10::irange(start, end)) {
                      auto self_i = index_data[i];
                      TORCH_CHECK_INDEX(
                          (self_i >= 0) && (self_i < self_dim_size),
                          "index out of range in self");
                      auto self_data = static_cast<char*>(selfSlice_data) +
                          self_i * self_stride_bytes;
                      auto result_data = static_cast<char*>(resultSlice_data) +
                          i * result_stride_bytes;
                      memcpy(result_data, self_data, slice_size_bytes);
                    }
                  });
            });
      } else {
        at::parallel_for(0, numel, grain_size / slice_size, outer_loop);
      }
    }
  } else {
    TORCH_CHECK(
        result.dim() <= 1,
        "result.dim() (",
        result.dim(),
        ") must one or zero for given self.dim() (",
        self.dim(),
        ")");
    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested
    // lambda
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "index_select",
        [&index_contig, &self, &result, &dim, &numel] {
          auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
          auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);

          auto self_data_ptr = self.data_ptr<scalar_t>();
          auto result_data_ptr = result.data_ptr<scalar_t>();
          auto self_numel = self.numel();
          AT_DISPATCH_INDEX_TYPES(
              index_contig.scalar_type(),
              "index_select_out_cpu_",
              [&index_contig,
               &numel,
               &self_numel,
               &self_data_ptr,
               &self_stride,
               &result_data_ptr,
               &result_stride] {
                auto index_data = index_contig.data_ptr<index_t>();
                for (const auto i : c10::irange(numel)) {
                  auto self_i = index_data[i];
                  TORCH_CHECK_INDEX(
                      (self_i >= 0) && (self_i < self_numel),
                      "index out of range in self");
                  scalar_t* self_ip = self_data_ptr + self_i * self_stride;
                  *(result_data_ptr + i * result_stride) = *self_ip;
                }
              });
        });
  }

  return result;
}

at::Tensor index_select_cpu_(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::index_select_cpu_\n");
#endif
  RECORD_FUNCTION(
      "torch_ipex::index_select_cpu_", c10::ArrayRef<c10::IValue>({}));

  at::Tensor result = at::empty({0}, self.options());
  return index_select_out_cpu_(self, dim, index, result);
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::index_select"),
      TORCH_FN((&torch_ipex::cpu::index_select_cpu_)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::index_select.out"),
      TORCH_FN((&torch_ipex::cpu::index_select_out_cpu_)));
}

} // namespace cpu
} // namespace torch_ipex