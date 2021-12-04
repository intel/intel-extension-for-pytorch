#pragma once

#include <ATen/ATen.h>

#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <algorithm>
#include <iostream>
#include "comm/Numerics.h"

using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <typename _Acc, typename _Size, typename _Value, typename _Compare>
_Size lower_bound_with_stride(
    _Acc acc,
    _Size first,
    _Size last,
    const _Value& value,
    _Compare comp,
    const size_t stride) {
  auto n = last - first;
  auto cur = n;
  _Size it;
  while (n > 0) {
    it = first;
    cur = n / stride / 2 * stride;
    it += cur;
    if (comp(acc[it], value)) {
      n -= cur + stride;
      it += stride;
      first = it;
    } else
      n = cur;
  }
  return first; // padding with stride
}

template <typename _Acc, typename _Size, typename _Value, typename _Compare>
inline _Size upper_bound_with_stride(
    _Acc acc,
    _Size first,
    _Size last,
    const _Value& value,
    _Compare comp,
    const size_t stride) {
  return lower_bound_with_stride(
      acc,
      first,
      last,
      value,
      [comp](auto x, auto y) { return !comp(y, x); },
      stride);
}

template <typename _IterKey, typename _IterValue, typename _Compare>
void bubble_sort_with_stride(
    _IterKey key,
    _IterValue value,
    const size_t begin,
    const size_t end,
    _Compare comp,
    const size_t stride) {
  if (begin < end) {
    for (size_t i = begin; i < end; i += stride) {
      for (size_t idx = i + stride; idx < end; idx += stride) {
        if (comp(key[idx], key[i])) {
          std::swap(key[i], key[idx]);
          std::swap(value[i], value[idx]);
        }
      }
    }
  }
}

template <
    typename _InKeyAcc,
    typename _InValueAcc,
    typename _OutKeyAcc,
    typename _OutValueAcc,
    typename _Compare>
void merge_with_stride(
    const size_t offset,
    _InKeyAcc& in_key_acc1,
    _InValueAcc& in_value_acc1,
    _OutKeyAcc& out_key_acc1,
    _OutValueAcc& out_value_acc1,
    const size_t start_1,
    const size_t end_1,
    const size_t end_2,
    const size_t start_out,
    _Compare comp,
    const size_t chunk,
    const size_t stride) {
  const size_t start_2 = end_1;
  // Borders of the sequences to merge within this call
  const size_t local_start_1 =
      DPCPP::min(static_cast<size_t>(offset + start_1), end_1);
  const size_t local_end_1 =
      DPCPP::min(static_cast<size_t>(local_start_1 + chunk * stride), end_1);
  const size_t local_start_2 =
      DPCPP::min(static_cast<size_t>(offset + start_2), end_2);
  const size_t local_end_2 =
      DPCPP::min(static_cast<size_t>(local_start_2 + chunk * stride), end_2);

  const size_t local_size_1 = local_end_1 - local_start_1;
  const size_t local_size_2 = local_end_2 - local_start_2;

  {
    // Process 1st sequence
    if (local_start_1 < local_end_1) {
      // Reduce the range for searching within the 2nd sequence and handle bound
      // items find left border in 2nd sequence
      const auto local_l_item_1 = in_key_acc1[local_start_1];
      const auto local_l_item_1_ = in_value_acc1[local_start_1];
      size_t l_search_bound_2 = lower_bound_with_stride(
          in_key_acc1, start_2, end_2, local_l_item_1, comp, stride);
      const size_t l_shift_1 = local_start_1 - start_1;
      const size_t l_shift_2 = l_search_bound_2 - start_2;

      out_key_acc1[start_out + l_shift_1 + l_shift_2] = local_l_item_1;
      out_value_acc1[start_out + l_shift_1 + l_shift_2] = local_l_item_1_;

      size_t r_search_bound_2{};
      // find right border in 2nd sequence
      if (local_size_1 > stride) {
        const auto local_r_item_1 = in_key_acc1[local_end_1 - stride];
        const auto local_r_item_1_ = in_value_acc1[local_end_1 - stride];
        r_search_bound_2 = lower_bound_with_stride(
            in_key_acc1, l_search_bound_2, end_2, local_r_item_1, comp, stride);
        const auto r_shift_1 = local_end_1 - stride - start_1;
        const auto r_shift_2 = r_search_bound_2 - start_2;

        out_key_acc1[start_out + r_shift_1 + r_shift_2] = local_r_item_1;
        out_value_acc1[start_out + r_shift_1 + r_shift_2] = local_r_item_1_;
      }

      // Handle intermediate items
      for (size_t idx = local_start_1 + stride; idx < local_end_1 - stride;
           idx += stride) {
        const auto intermediate_item_1 = in_key_acc1[idx];
        const auto intermediate_item_1_ = in_value_acc1[idx];
        // we shouldn't seek in whole 2nd sequence. Just for the part where the
        // 1st sequence should be
        l_search_bound_2 = lower_bound_with_stride(
            in_key_acc1,
            l_search_bound_2,
            r_search_bound_2,
            intermediate_item_1,
            comp,
            stride);
        const size_t shift_1 = idx - start_1;
        const size_t shift_2 = l_search_bound_2 - start_2;
        out_key_acc1[start_out + shift_1 + shift_2] = intermediate_item_1;
        out_value_acc1[start_out + shift_1 + shift_2] = intermediate_item_1_;
      }
    }
    // Process 2nd sequence
    if (local_start_2 < local_end_2) {
      // Reduce the range for searching within the 1st sequence and handle bound
      // items find left border in 1st sequence
      const auto local_l_item_2 = in_key_acc1[local_start_2];
      const auto local_l_item_2_ = in_value_acc1[local_start_2];
      size_t l_search_bound_1 = upper_bound_with_stride(
          in_key_acc1, start_1, end_1, local_l_item_2, comp, stride);
      const size_t l_shift_1 = l_search_bound_1 - start_1;
      const size_t l_shift_2 = local_start_2 - start_2;

      out_key_acc1[start_out + l_shift_1 + l_shift_2] = local_l_item_2;
      out_value_acc1[start_out + l_shift_1 + l_shift_2] = local_l_item_2_;

      size_t r_search_bound_1{};
      // find right border in 1st sequence
      if (local_size_2 > stride) {
        const auto local_r_item_2 = in_key_acc1[local_end_2 - stride];
        const auto local_r_item_2_ = in_value_acc1[local_end_2 - stride];
        r_search_bound_1 = upper_bound_with_stride(
            in_key_acc1, l_search_bound_1, end_1, local_r_item_2, comp, stride);
        const size_t r_shift_1 = r_search_bound_1 - start_1;
        const size_t r_shift_2 = local_end_2 - stride - start_2;

        out_key_acc1[start_out + r_shift_1 + r_shift_2] = local_r_item_2;
        out_value_acc1[start_out + r_shift_1 + r_shift_2] = local_r_item_2_;
      }

      // Handle intermediate items
      for (auto idx = local_start_2 + stride; idx < local_end_2 - stride;
           idx += stride) {
        const auto intermediate_item_2 = in_key_acc1[idx];
        const auto intermediate_item_2_ = in_value_acc1[idx];
        // we shouldn't seek in whole 1st sequence. Just for the part where the
        // 2nd sequence should be
        l_search_bound_1 = upper_bound_with_stride(
            in_key_acc1,
            l_search_bound_1,
            r_search_bound_1,
            intermediate_item_2,
            comp,
            stride);
        const size_t shift_1 = l_search_bound_1 - start_1;
        const size_t shift_2 = idx - start_2;
        out_key_acc1[start_out + shift_1 + shift_2] = intermediate_item_2;
        out_value_acc1[start_out + shift_1 + shift_2] = intermediate_item_2_;
      }
    }
  }
}

template <
    typename _Group,
    typename _IterKey,
    typename _IterValue,
    typename _Compare,
    typename _Id>
void merge_sort_with_stride(
    _Group group,
    _IterKey key,
    _IterValue value,
    const size_t size, // element wise range
    _Compare comp,
    uint8_t* scratch_key,
    uint8_t* scratch_value,
    _Id id,
    const size_t stride = 1) {
  using TK = typename std::iterator_traits<_IterKey>::value_type;
  using TV = typename std::iterator_traits<_IterValue>::value_type;
  const size_t local = group.get_local_range(0);
  const size_t idx = id.get_local_id();
  const size_t chunk = (size / stride - 1) / local + 1;
  const size_t chunk_times_stride = chunk * stride;

  // we need to sort within work item firstly
  bubble_sort_with_stride(
      key,
      value,
      idx * chunk_times_stride,
      DPCPP::min((idx + 1) * chunk_times_stride, size),
      comp,
      stride);
  id.barrier();

  TK* temp_key = reinterpret_cast<TK*>(scratch_key);
  TV* temp_value = reinterpret_cast<TV*>(scratch_value);

  bool data_in_temp = false;
  size_t sorted_size = 1;
  while (sorted_size * chunk_times_stride < size) {
    const size_t start_1 = DPCPP::min(
        2 * sorted_size * chunk_times_stride * (idx / sorted_size),
        size); // TODO
    const size_t end_1 =
        DPCPP::min(start_1 + sorted_size * chunk_times_stride, size);
    const size_t start_2 = end_1;
    const size_t end_2 =
        DPCPP::min(start_2 + sorted_size * chunk_times_stride, size);
    const size_t offset = chunk_times_stride * (idx % sorted_size);

    if (!data_in_temp) {
      merge_with_stride(
          offset,
          key,
          value,
          temp_key,
          temp_value,
          start_1,
          end_1,
          end_2,
          start_1,
          comp,
          chunk,
          stride);
    } else {
      merge_with_stride(
          offset,
          temp_key,
          temp_value,
          key,
          value,
          start_1,
          end_1,
          end_2,
          start_1,
          comp,
          chunk,
          stride);
    }
    id.barrier();

    data_in_temp = !data_in_temp;
    sorted_size *= 2;
  }

  // copy back if data is in a temporary storage
  if (data_in_temp) {
    for (size_t i = 0; i < chunk_times_stride; i += stride) {
      if (idx * chunk_times_stride + i < size) {
        key[idx * chunk_times_stride + i] =
            temp_key[idx * chunk_times_stride + i];
        value[idx * chunk_times_stride + i] =
            temp_value[idx * chunk_times_stride + i];
      }
    }
    id.barrier();
  }
}

template <typename _Compare = std::less<>>
class merge_sorter {
  _Compare comp;
  uint8_t* scratch_key;
  uint8_t* scratch_value;
  size_t scratch_key_size;
  size_t scratch_value_size;
  size_t stride;
  DPCPP::nd_item<1> id;

 public:
  template <std::size_t Extent>
  merge_sorter(
      DPCPP::span<uint8_t, Extent> scratch_key_,
      DPCPP::span<uint8_t, Extent> scratch_value_,
      DPCPP::nd_item<1> id_,
      _Compare comp_ = _Compare(),
      const size_t stride_ = 1)
      : comp(comp_),
        scratch_key(scratch_key_.data()),
        scratch_value(scratch_value_.data()),
        scratch_key_size(scratch_key_.size()),
        scratch_value_size(scratch_value_.size()),
        stride(stride_),
        id(id_) {}

  template <typename _Group, typename _PtrKey, typename _PtrValue>
  void operator()(
      _Group g,
      _PtrKey key_begin,
      _PtrKey key_end,
      _PtrValue value_begin,
      _PtrValue value_end) {
    using TK = typename std::iterator_traits<_PtrKey>::value_type;
    using TV = typename std::iterator_traits<_PtrValue>::value_type;
    merge_sort_with_stride(
        g,
        key_begin,
        value_begin,
        key_end - key_begin,
        comp,
        scratch_key,
        scratch_value,
        id,
        stride);
  }
};

template <
    typename _Group,
    typename _IterKey,
    typename _IterValue,
    typename _Compare,
    std::size_t Extent>
void merge_sort(
    _Group g,
    DPCPP::span<uint8_t, Extent> scratch_key,
    DPCPP::span<uint8_t, Extent> scratch_value,
    _IterKey key_first,
    _IterKey key_last,
    _IterValue value_first,
    _IterValue value_last,
    _Compare comp,
    DPCPP::nd_item<1> id,
    const size_t stride = 1) {
  auto sorter =
      merge_sorter<_Compare>(scratch_key, scratch_value, id, comp, stride);
  sorter(g, key_first, key_last, value_first, value_last);
}

} // namespace impl

#define MERGE_SORT_EXEC(COMP)                               \
  impl::merge_sort(                                         \
      id.get_group(),                                       \
      DPCPP::span{                                          \
          &scratch_key[start * element_size_of_key],        \
          sort_item_size * stride * element_size_of_key},   \
      DPCPP::span{                                          \
          &scratch_value[start * element_size_of_value],    \
          sort_item_size * stride * element_size_of_value}, \
      keys,                                                 \
      keys + sort_item_size * stride,                       \
      vals,                                                 \
      vals + sort_item_size * stride,                       \
      COMP,                                                 \
      id,                                                   \
      stride);

template <typename key_t, typename value_t>
void merge_sort_kernel(
    key_t* key,
    value_t* value,
    const size_t sort_group_size,
    const size_t sort_item_size,
    const bool dir,
    uint8_t* scratch_key,
    uint8_t* scratch_value,
    const int stride = 1) {
  auto& q = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto max_group_size = dpcppMaxWorkGroupSize(dev_id);
  auto element_size_of_key = sizeof(key_t);
  auto element_size_of_value = sizeof(value_t);

  auto cgf = DPCPP_Q_CGF(h) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> id) {
      auto group_id = id.get_group(0);
      auto start =
          group_id / stride * stride * sort_item_size + group_id % stride;
      auto keys = &key[start];
      auto vals = &value[start];
      if (dir) {
        MERGE_SORT_EXEC(
            [](auto x, auto y) { return std::greater<key_t>()(x, y); })
      } else {
        MERGE_SORT_EXEC([](auto x, auto y) { return std::less<key_t>()(x, y); })
      }
    };
    h.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(sort_group_size * max_group_size),
            DPCPP::range<1>(max_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

template <typename key_t, typename value_t, class comp_t>
void merge_sort_kernel(
    key_t* key,
    value_t* value,
    const size_t sort_group_size,
    const size_t sort_item_size,
    uint8_t* scratch_key,
    uint8_t* scratch_value,
    const comp_t comp_op,
    const int stride = 1) {
  auto& q = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto max_group_size = dpcppMaxWorkGroupSize(dev_id);
  auto element_size_of_key = sizeof(key_t);
  auto element_size_of_value = sizeof(value_t);

  auto cgf = DPCPP_Q_CGF(h) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> id) {
      auto group_id = id.get_group(0);
      auto start =
          group_id / stride * stride * sort_item_size + group_id % stride;
      auto keys = &key[start];
      auto vals = &value[start];
      MERGE_SORT_EXEC([=](auto x, auto y) { return comp_op(x, y); })
    };
    h.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(sort_group_size * max_group_size),
            DPCPP::range<1>(max_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

} // namespace AtenIpexTypeXPU
} // namespace at
