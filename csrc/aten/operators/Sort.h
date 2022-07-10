#pragma once

#include <ATen/MemoryOverlap.h>
#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include <algorithm>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include "SortingDeviceRadixSort.h"
#include "SortingGroupRadixSort.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <typename key_t, typename value_t, typename func_t>
inline void host_kvsort(
    key_t* kbegin,
    key_t* kend,
    value_t* vbegin,
    const func_t& fn) {
  for (auto kit = kbegin, vit = vbegin; kit != kend; kit++, vit++) {
    for (auto kit_ = kit, vit_ = vit; kit_ != kend; kit_++, vit_++) {
      if (fn(*kit, *kit_)) {
        std::swap(*kit, *kit_);
        std::swap(*vit, *vit_);
      }
    }
  }
}

std::vector<int64_t> infer_dense_strides_dim_last(
    const Tensor& self,
    int64_t dim) {
  int64_t ndim = self.dim();
  // sort the strides in descending order according to its value,
  // keeping dim the last.
  std::vector<int64_t> strides = self.strides().vec();
  strides[dim] = -1;
  std::vector<int64_t> original_dim(ndim);
  for (int64_t i = 0; i < ndim; i++) {
    original_dim[i] = i;
  }
  host_kvsort(
      strides.data(),
      strides.data() + ndim,
      original_dim.data(),
      std::greater<int64_t>());
  // generate contiguous strides on permuted dims
  std::vector<int64_t> new_strides(ndim);
  std::vector<int64_t> new_strides_unsort(ndim);
  int64_t cumprod = 1;
  for (int64_t i = 0; i < ndim; i++) {
    new_strides[ndim - 1 - i] = cumprod;
    cumprod *= self.sizes()[original_dim[ndim - 1 - i]];
  }
  // unsort new strides
  for (int64_t i = 0; i < ndim; i++) {
    new_strides_unsort[original_dim[i]] = new_strides[i];
  }
  return new_strides_unsort;
}

} // namespace impl

std::tuple<Tensor&, Tensor&> sort_out_stable(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  TORCH_INTERNAL_ASSERT(
      stable.has_value(),
      "sort_out(): c10::optional<bool> for stable has to have value.");

  bool is_non_overlapping_and_dense = self.is_non_overlapping_and_dense();
  int64_t numel = self.numel();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nsort = self.sizes()[dim];

  TORCH_CHECK(
      nsort <= std::numeric_limits<int>::max(),
      "The dimension being sorted can not have more than INT_MAX elements.");

  const auto self_dtype = self.dtype();
  TORCH_CHECK(
      self_dtype != ScalarType::ComplexFloat &&
          self_dtype != ScalarType::ComplexDouble,
      "Sort currently does not support complex dtypes on XPU.");

  if (ndim == 0) {
    if (!values.defined()) {
      values = self.clone();
    } else {
      values.resize_as_(self);
      values.copy_(self);
    }
    if (!indices.defined()) {
      indices = at::zeros({}, self.options().dtype(kLong));
    } else {
      indices.resize_as_(self);
      indices.zero_();
    }
    return std::forward_as_tuple(values, indices);
  }

  Tensor self_;
  bool newself = false;
  if (is_non_overlapping_and_dense && self.stride(dim) == 1) {
    self_ = self;
  } else {
    auto new_strides_unsort = impl::infer_dense_strides_dim_last(self, dim);
    self_ = at::empty_strided(self.sizes(), new_strides_unsort, self.options());
    self_.copy_(self);
    newself = true;
  }

  Tensor values_tmp, indices_tmp;
  void* values_ptr_;
  int64_t* indices_ptr;
  if (!values.defined()) {
    if (is_non_overlapping_and_dense) {
      values = at::empty_strided(self.sizes(), self.strides(), self.options());
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      values = at::empty_strided(self.sizes(), strides, self.options());
    }
  } else {
    TORCH_CHECK(
        self_.scalar_type() == values.scalar_type(),
        "Unexpected dtype for values, expect ",
        self_.scalar_type(),
        ", got ",
        values.scalar_type());
    values.resize_as_(self);
  }

  if (values.strides() == self_.strides() &&
      (newself || get_overlap_status(self, values) == MemOverlapStatus::NO)) {
    values_ptr_ = values.data_ptr();
  } else {
    values_tmp =
        at::empty_strided(self_.sizes(), self_.strides(), self_.options());
    values_ptr_ = values_tmp.data_ptr();
  }

  if (!indices.defined()) {
    if (is_non_overlapping_and_dense) {
      indices = at::empty_strided(
          self.sizes(), self.strides(), self.options().dtype(kLong));
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      indices =
          at::empty_strided(self.sizes(), strides, self.options().dtype(kLong));
    }
  } else {
    TORCH_CHECK(
        kLong == indices.scalar_type(),
        "Unexpected dtype for values, expect torch.long, got ",
        indices.scalar_type());
    indices.resize_as_(self);
  }

  if (indices.strides() != self_.strides()) {
    indices_tmp = at::empty_strided(
        self_.sizes(), self_.strides(), self_.options().dtype(kLong));
    indices_ptr = indices_tmp.data_ptr<int64_t>();
  } else {
    indices_ptr = indices.data_ptr<int64_t>();
  }

  if (numel == 0) {
    return std::forward_as_tuple(values, indices);
  }

  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self_.scalar_type(),
      "segmented_merge_sort_kernel",
      [&]() {
        scalar_t* self_ptr = self_.data_ptr<scalar_t>();
        using offset_t = uint32_t;
        auto info = getTensorInfo<scalar_t, offset_t>(self_);
        info.reduceDim(dim);
        int collapse_dim = info.collapseDims(dim);
        int nsegments = 1;
        for (int i = 0; i < info.dims; ++i)
          nsegments *= info.sizes[i];
        int stride = info.strides[collapse_dim];

        SegmentedGroupRadixSortDesc desc(
            nsegments, nsort, stride, descending, true);
        if (desc.valid()) {
          if (!desc.need_temp()) {
            segmented_group_radix_sort_kernel<
                scalar_t,
                int64_t,
                uint16_t,
                true>(
                desc,
                self_ptr,
                (scalar_t*)values_ptr_,
                nullptr,
                (int64_t*)indices_ptr,
                [=](offset_t slice) -> offset_t {
                  return IndexToOffset<scalar_t, offset_t>::get(slice, info);
                });
          } else {
            auto sorting_tmp_k = at::empty_strided(
                self_.sizes(), self_.strides(), self_.options());
            auto sorting_tmp_v = at::empty_strided(
                self_.sizes(), self_.strides(), self_.options().dtype(kLong));
            segmented_group_radix_sort_kernel<
                scalar_t,
                int64_t,
                uint16_t,
                true>(
                desc,
                self_ptr,
                (scalar_t*)values_ptr_,
                nullptr,
                (int64_t*)indices_ptr,
                [=](offset_t slice) -> offset_t {
                  return IndexToOffset<scalar_t, offset_t>::get(slice, info);
                },
                (scalar_t*)sorting_tmp_k.data_ptr(),
                (int64_t*)sorting_tmp_v.data_ptr());
          }
        } else {
          SegmentedDeviceRadixSortDesc desc(
              nsegments, nsort, stride, descending, true);
          auto sorting_tmp_k = at::empty_strided(
              self_.sizes(), self_.strides(), self_.options());
          auto sorting_tmp_v = at::empty_strided(
              self_.sizes(), self_.strides(), self_.options().dtype(kLong));
          segmented_device_radix_sort_kernel<scalar_t, int64_t>(
              desc,
              self_ptr,
              (scalar_t*)values_ptr_,
              nullptr,
              (int64_t*)indices_ptr,
              [=](offset_t slice) -> offset_t {
                return IndexToOffset<scalar_t, offset_t>::get(slice, info);
              },
              (scalar_t*)sorting_tmp_k.data_ptr(),
              (int64_t*)sorting_tmp_v.data_ptr());
        }
      });

  if (values_tmp.defined()) {
    values.copy_(values_tmp);
  }
  if (indices_tmp.defined()) {
    indices.copy_(indices_tmp);
  }
  return std::forward_as_tuple(values, indices);
}

} // namespace AtenIpexTypeXPU
} // namespace at
