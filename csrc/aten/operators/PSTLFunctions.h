#pragma once

#include <ATen/record_function.h>
#include <utils/DPCPP.h>
#include "BitonicMergeSort.h"
#include "Scan.h"
#include "SortingDeviceRadixSort.h"
#include "SortingGroupRadixSort.h"
#include "comm/Atomics.h"
#include "comm/KeyTraits.h"
#include "comm/MathReduce.h"
#include "comm/SimpleReduce.h"

using namespace at::AtenIpexTypeXPU;
namespace xpu {
namespace pstl {
template <int scan_type, class InputIt, class OutputIt, class T>
DPCPP_DEVICE static inline OutputIt _scan_kernel(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  auto options = map_options<T>();

  if (N <= wgroup_size) {
    // Kogge-Stone addr algorithm;
    auto cgf = DPCPP_Q_CGF(__cgh) {
      dpcpp_local_acc_t<T> local_scan(N, __cgh);

      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
        auto local_id = item_id.get_local_linear_id();

        // initialize local_input
        auto cur_init = init;
        if (scan_type == 1) {
          local_scan[local_id] = first[local_id];
        } else {
          if (local_id > 0)
            local_scan[local_id] = first[local_id - 1];
          else
            local_scan[local_id] = 0;
        }
        if (local_id == 0)
          local_scan[local_id] += cur_init;
        item_id.barrier(dpcpp_local_fence);

        // body of KS algo
        for (auto __k = 1; __k < N; __k <<= 1) {
          auto tmp = (local_id >= __k) ? local_scan[local_id - __k] : 0;
          item_id.barrier(dpcpp_local_fence);
          local_scan[local_id] += tmp;
          item_id.barrier(dpcpp_local_fence);
        }

        // flush result into dst
        d_first[local_id] = local_scan[local_id];
      };
      __cgh.parallel_for(sycl::nd_range</*dim=*/1>(N, N), kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return d_first + N;
  }

  Tensor carry = at::empty({ngroups}, options);
  T* carry_ptr = carry.data_ptr<T>();

  // 1. do exclusive_scan on each workgroups
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    dpcpp_local_acc_t<T> local_scan(wgroup_size, __cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      auto local_id = item_id.get_local_linear_id();
      auto global_id = item_id.get_global_linear_id();
      auto group_id = item_id.get_group_linear_id();

      // initialize local_input
      auto cur_init = (group_id == 0 ? init : 0);
      if (global_id < N) {
        if (scan_type == 1) {
          local_scan[local_id] = first[global_id];
        } else {
          if (local_id > 0)
            local_scan[local_id] = first[global_id - 1];
          else
            local_scan[local_id] = 0;
        }
        if (local_id == 0)
          local_scan[local_id] += cur_init;
        if (local_id == wgroup_size - 1) {
          carry_ptr[group_id] = first[global_id];
        }
      }
      item_id.barrier(dpcpp_local_fence);

      // body of KS algo
      for (auto __k = 1; __k < wgroup_size; __k <<= 1) {
        auto tmp = (local_id >= __k) ? local_scan[local_id - __k] : 0;
        item_id.barrier(dpcpp_local_fence);
        local_scan[local_id] += tmp;
        item_id.barrier(dpcpp_local_fence);
      }

      // flush result into dst
      if (global_id < N) {
        d_first[global_id] = local_scan[local_id];
      }
      if (local_id == wgroup_size - 1) {
        if (scan_type == 1)
          carry_ptr[group_id] = local_scan[local_id];
        else
          carry_ptr[group_id] += local_scan[local_id];
      }
    };

    __cgh.parallel_for(
        sycl::nd_range</*dim=*/1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. recursion for carry
  _scan_kernel<0>(carry_ptr, carry_ptr + ngroups, carry_ptr, (T)0);

  // 3. reduce among all work groups and flush data to dst
  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    dpcpp_local_acc_t<T> local_carry(1, __cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      auto local_id = item_id.get_local_linear_id();
      auto global_id = item_id.get_global_linear_id();
      auto group_id = item_id.get_group_linear_id();

      if (local_id == 0)
        local_carry[0] = carry_ptr[group_id];
      item_id.barrier(dpcpp_local_fence);

      if (global_id < N) {
        d_first[global_id] += local_carry[0];
      }
    };
    __cgh.parallel_for(
        sycl::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  return d_first + N;
}

template <typename T, class InputIt, class OutputIt>
DPCPP_DEVICE static inline OutputIt exclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  RECORD_FUNCTION("exclusive_scan_xpu", {});
  return _scan_kernel<0>(first, last, d_first, init);
}

template <typename T, class InputIt, class OutputIt>
DPCPP_DEVICE static inline OutputIt inclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  RECORD_FUNCTION("inclusive_scan_xpu", {});
  return _scan_kernel<1>(first, last, d_first, init);
}

template <typename index_t, class InputIt, class OutputIt, class UnaryPredicate>
DPCPP_DEVICE static inline OutputIt copy_if(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    UnaryPredicate pred) {
  RECORD_FUNCTION("copy_if_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      if (first) {
        gmask_ptr[item_id] =
            static_cast<index_t>(static_cast<bool>(pred(first[item_id])));
      } else {
        gmask_ptr[item_id] =
            static_cast<index_t>(static_cast<bool>(pred(item_id)));
      }
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. get target positions(with shift -1) using inclusive_scan
  inclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      if (gmask_ptr[item_id] != 0) {
        if (first) {
          d_first[tpos_ptr[item_id] - /*inclusive shift*/ 1] = first[item_id];
        } else {
          d_first[tpos_ptr[item_id] - /*inclusive shift*/ 1] = item_id;
        }
      }
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  index_t M = target_pos[N - 1].template item<index_t>();
  return d_first + M;
}

template <
    typename output_t,
    class InputIt,
    class OutputIt,
    class UnaryOperation>
DPCPP_DEVICE static inline OutputIt transform(
    InputIt first1,
    InputIt last1,
    OutputIt d_first,
    UnaryOperation unary_op) {
  RECORD_FUNCTION("transform_1_xpu", {});
  const auto N = std::distance(first1, last1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      d_first[item_id] = static_cast<output_t>(unary_op(first1[item_id]));
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return d_first + N;
}

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class OutputIt,
    class BinaryOperation>
DPCPP_DEVICE static inline OutputIt transform(
    InputIt1 first1,
    InputIt1 last1,
    InputIt2 first2,
    OutputIt d_first,
    BinaryOperation binary_op) {
  RECORD_FUNCTION("transform_2_xpu", {});
  const auto N = std::distance(first1, last1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      d_first[item_id] =
          static_cast<output_t>(binary_op(first1[item_id], first2[item_id]));
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return d_first + N;
}

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class OutputIt,
    class BinaryOperation>
DPCPP_DEVICE static inline OutputIt transform_first_true(
    InputIt1 first1,
    InputIt1 last1,
    InputIt2 first2,
    OutputIt d_first,
    BinaryOperation binary_op) {
  RECORD_FUNCTION("transform_first_true", {});
  const auto N = std::distance(first1, last1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      first1[0] = 1;
      d_first[item_id] =
          static_cast<output_t>(binary_op(first1[item_id], first2[item_id]));
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return d_first + N;
}

template <class T, class ForwardIt>
DPCPP_DEVICE static inline void iota(ForwardIt first, ForwardIt last, T value) {
  RECORD_FUNCTION("iota_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      first[item_id] = value + static_cast<T>(item_id);
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T, typename index_t, class ForwardIt, class BinaryPredicate>
ForwardIt unique(ForwardIt first, ForwardIt last, BinaryPredicate p) {
  RECORD_FUNCTION("unique_kernel_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto options = map_options<T>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      if (item_id > 0)
        gmask_ptr[item_id] = static_cast<index_t>(
            static_cast<bool>(!p(first[item_id - 1], first[item_id])));
      else
        gmask_ptr[item_id] = static_cast<index_t>(1);
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. get target positions with exclusive_scan
  exclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  Tensor scratchpad = at::empty({N}, options);
  T* scratchpad_ptr = scratchpad.data_ptr<T>();

  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      if (gmask_ptr[item_id] != 0)
        scratchpad_ptr[tpos_ptr[item_id]] = first[item_id];
    };

    __cgh.parallel_for(sycl::range</*dim=*/>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  index_t M = global_mask[N - 1].template item<index_t>() +
      target_pos[N - 1].template item<index_t>();

  auto cgf_4 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      first[item_id] = scratchpad_ptr[item_id];
    };

    __cgh.parallel_for(sycl::range</*dim=*/>(M), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_4);

  return first + M;
}

template <
    typename T,
    typename zT,
    typename index_t,
    class ForwardIt,
    class ZipForwardIt,
    class BinaryPredicate>
std::tuple<ForwardIt, ZipForwardIt> unique_with_zip(
    ForwardIt first,
    ForwardIt last,
    ZipForwardIt z_first,
    BinaryPredicate p) {
  RECORD_FUNCTION("unique_with_zip_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto options = map_options<T>();
  auto z_options = map_options<zT>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      if (item_id > 0)
        gmask_ptr[item_id] = static_cast<index_t>(
            static_cast<bool>(!p(first[item_id - 1], first[item_id])));
      else
        gmask_ptr[item_id] = static_cast<index_t>(1);
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. get target positions with exclusive_scan
  exclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  Tensor scratchpad = at::empty({N}, options);
  Tensor z_scratchpad = at::empty({N}, z_options);
  T* scratchpad_ptr = scratchpad.data_ptr<T>();
  zT* z_scratchpad_ptr = z_scratchpad.data_ptr<zT>();

  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      if (gmask_ptr[item_id] != 0) {
        scratchpad_ptr[tpos_ptr[item_id]] = first[item_id];
        z_scratchpad_ptr[tpos_ptr[item_id]] = z_first[item_id];
      }
    };

    __cgh.parallel_for(sycl::range</*dim=*/>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  index_t M = global_mask[N - 1].template item<index_t>() +
      target_pos[N - 1].template item<index_t>();

  auto cgf_4 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      first[item_id] = scratchpad_ptr[item_id];
      z_first[item_id] = z_scratchpad_ptr[item_id];
    };

    __cgh.parallel_for(sycl::range</*dim=*/>(M), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_4);

  return std::make_tuple<ForwardIt, ZipForwardIt>(first + M, z_first + M);
}

template <
    typename output_t,
    class InputIt,
    class OutputIt,
    class BinaryOperation>
OutputIt adjacent_difference(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    BinaryOperation op) {
  RECORD_FUNCTION("adjacent_difference", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  Tensor scratchpad;
  OutputIt adiff = d_first;
  bool is_inplace = (void*)first == (void*)d_first ? true : false;
  if (is_inplace) {
    scratchpad = at::empty({N}, map_options<output_t>());
    adiff = scratchpad.data_ptr<output_t>();
  }

  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      if (item_id > 0)
        adiff[item_id] =
            static_cast<output_t>(op(first[item_id - 1], first[item_id]));
      else
        adiff[item_id] = static_cast<output_t>(first[item_id]);
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  if (is_inplace) {
    auto cgf_2 = DPCPP_Q_CGF(__cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
        d_first[item_id] = adiff[item_id];
      };

      __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf_2);
  }

  return d_first + N;
}

template <typename output_t, class InputIt, class OutputIt>
OutputIt adjacent_difference(InputIt first, InputIt last, OutputIt d_first) {
  return adjacent_difference<output_t>(
      first, last, d_first, [](auto l, auto r) { return r - l; });
}

template <
    typename input_t,
    typename output_t,
    typename index_t,
    class InputIt,
    class OutputIt,
    class BinaryPredicate>
OutputIt count_by_segment(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    BinaryPredicate p) {
  RECORD_FUNCTION("count_by_segment_xpu", {});
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto options = map_options<output_t>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      if (item_id > 0)
        gmask_ptr[item_id] = static_cast<index_t>(
            static_cast<bool>(!p(first[item_id - 1], first[item_id])));
      else
        gmask_ptr[item_id] = static_cast<index_t>(1);
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. get target positions with inclusive_scan
  inclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. calculate counts for each unique point
  Tensor range = at::empty({N + 1}, options);
  output_t* range_ptr = range.data_ptr<output_t>();
  auto range_begin = range_ptr;
  iota(range_begin, range_begin + N + 1, (output_t)0);
  Tensor picked_range = at::empty({N + 1}, options);
  output_t* picked_range_ptr = picked_range.data_ptr<output_t>();
  auto picked_range_begin = picked_range_ptr;
  auto picked_range_end = picked_range_begin;
  picked_range_end = copy_if<index_t>(
      range_begin, range_begin + N, picked_range_begin, [=](output_t a) {
        return gmask_ptr[a] != 0;
      });
  auto num_out = std::distance(picked_range_begin, picked_range_end);
  picked_range[num_out] = N;
  // notice: the temp tensor `range` will be re-used to store the result of
  // adjacent_difference
  adjacent_difference<index_t>(
      picked_range_begin + 1, picked_range_begin + num_out + 1, range_begin);

  // 4. flush range to every elements of counts
  auto cgf_4 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      d_first[item_id] = range_ptr[tpos_ptr[item_id]];
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(N), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_4);

  return d_first + N;
}

// bubble sort for the first round sorting
template <typename KeyType, typename ValueType, typename CompFunc>
inline void leaf_sort(
    const sycl::item<1>& item,
    KeyType* key,
    ValueType* val,
    size_t n,
    size_t sorted_sz,
    const CompFunc& comp_t) {
  auto start = item.get_linear_id() * n;
  auto end = std::min(start + n, sorted_sz);
  for (size_t i = start; i < end; ++i) {
    for (size_t j = start + 1; j < start + end - i; ++j) {
      // for stable sort, the condition should be:
      // if comp_t(key[j], key[j-1]), swap two elements;
      // so when key[j]==key[j-1], no swap.
      at::AtenIpexTypeXPU::impl::compare_and_swap(
          key[j], val[j], key[j - 1], val[j - 1], true, comp_t);
    }
  }
}

// lower_bound used in merge sort: pick up the elements in the sequence doesn't
// meet the compare situation with smallest index
template <typename KeyType, typename CompFunc>
inline size_t lower_bound(
    KeyType* in_data,
    size_t first,
    size_t last,
    const KeyType& key,
    const CompFunc& comp_t) {
  auto n = last - first;
  auto cur = n;
  size_t it;
  while (n > 0) {
    it = first;
    cur = n / 2;
    it += cur;
    if (comp_t(in_data[it], key)) {
      n -= cur + 1;
      first = ++it;
    } else {
      n = cur;
    }
  }
  return first;
}

template <typename KeyType, typename CompFunc>
inline size_t upper_bound(
    KeyType* in_data,
    size_t first,
    size_t last,
    const KeyType& key,
    const CompFunc& comp_t) {
  auto n = last - first;
  auto cur = n;
  size_t it;
  while (n > 0) {
    it = first;
    cur = n / 2;
    it += cur;
    if (!comp_t(key, in_data[it])) {
      n -= cur + 1;
      first = ++it;
    } else {
      n = cur;
    }
  }
  return first;
}

template <typename KeyType, typename ValueType, typename CompFunc>
inline void merge(
    const size_t offset,
    KeyType* in_key,
    ValueType* in_val,
    KeyType* out_key,
    ValueType* out_val,
    const size_t sq1_start,
    const size_t sq1_end,
    const size_t sq2_start,
    const size_t sq2_end,
    const size_t chunk_size,
    const CompFunc& comp_t) {
  const size_t chunk1_start = std::min((offset + sq1_start), sq1_end);
  const size_t chunk1_end = std::min((chunk1_start + chunk_size), sq1_end);
  const size_t chunk2_start = std::min((offset + sq2_start), sq2_end);
  const size_t chunk2_end = std::min((chunk2_start + chunk_size), sq2_end);

  const size_t chunk1_size = chunk1_end - chunk1_start;
  const size_t chunk2_size = chunk2_end - chunk2_start;

  size_t l_sq2_low_bound;
  size_t r_sq2_low_bound;
  size_t l_sq1_upper_bound;
  size_t r_sq1_upper_bound;
  if (!comp_t(in_key[sq2_start], in_key[sq1_end - 1])) {
    for (unsigned int i = 0; i < chunk1_size; ++i) {
      out_key[chunk1_start + i] = in_key[chunk1_start + i];
      out_val[chunk1_start + i] = in_val[chunk1_start + i];
    }

    for (unsigned int i = 0; i < chunk2_size; ++i) {
      out_key[chunk2_start + i] = in_key[chunk2_start + i];
      out_val[chunk2_start + i] = in_val[chunk2_start + i];
    }
  } else if (!comp_t(in_key[sq1_start], in_key[sq2_end - 1])) {
    auto out1_offset = sq2_end - sq2_start + chunk1_start;
    auto out2_offset = sq1_start + chunk2_start - sq2_start;
    for (unsigned int i = 0; i < chunk1_size; ++i) {
      out_key[out1_offset + i] = in_key[chunk1_start + i];
      out_val[out1_offset + i] = in_val[chunk1_start + i];
    }

    for (unsigned int i = 0; i < chunk2_size; ++i) {
      out_key[out2_offset + i] = in_key[chunk2_start + i];
      out_val[out2_offset + i] = in_val[chunk2_start + i];
    }
  } else {
    // Process 1st sequence
    if (chunk1_start < chunk1_end) {
      const auto chunk1_l_item = in_key[chunk1_start];
      l_sq2_low_bound =
          lower_bound(in_key, sq2_start, sq2_end, chunk1_l_item, comp_t);
      const auto l_shift1 = chunk1_start - sq1_start;
      const auto l_shift2 = l_sq2_low_bound - sq2_start;
      out_key[sq1_start + l_shift1 + l_shift2] = chunk1_l_item;
      out_val[sq1_start + l_shift1 + l_shift2] = in_val[chunk1_start];
      if (chunk1_end - chunk1_start > 1) {
        const auto chunk1_r_item = in_key[chunk1_end - 1];
        r_sq2_low_bound = lower_bound(
            in_key, l_sq2_low_bound, sq2_end, chunk1_r_item, comp_t);
        const auto r_shift1 = chunk1_end - 1 - sq1_start;
        const auto r_shift2 = r_sq2_low_bound - sq2_start;
        out_key[sq1_start + r_shift1 + r_shift2] = chunk1_r_item;
        out_val[sq1_start + r_shift1 + r_shift2] = in_val[chunk1_end - 1];
      }
      for (auto idx = chunk1_start + 1; idx < chunk1_end - 1; ++idx) {
        const auto inter_item_1 = in_key[idx];
        l_sq2_low_bound = lower_bound(
            in_key, l_sq2_low_bound, r_sq2_low_bound, inter_item_1, comp_t);
        const auto shift1 = idx - sq1_start;
        const auto shift2 = l_sq2_low_bound - sq2_start;
        out_key[sq1_start + shift1 + shift2] = inter_item_1;
        out_val[sq1_start + shift1 + shift2] = in_val[idx];
      }
    }
    // Process 2nd sequence
    if (chunk2_start < chunk2_end) {
      const auto chunk2_l_item = in_key[chunk2_start];
      l_sq1_upper_bound =
          upper_bound(in_key, sq1_start, sq1_end, chunk2_l_item, comp_t);
      const auto l_shift1 = l_sq1_upper_bound - sq1_start;
      const auto l_shift2 = chunk2_start - sq2_start;
      out_key[sq1_start + l_shift1 + l_shift2] = chunk2_l_item;
      out_val[sq1_start + l_shift1 + l_shift2] = in_val[chunk2_start];
      if (chunk2_end - chunk2_start > 1) {
        const auto chunk2_r_item = in_key[chunk2_end - 1];
        r_sq1_upper_bound = upper_bound(
            in_key, l_sq1_upper_bound, sq1_end, chunk2_r_item, comp_t);
        const auto r_shift1 = r_sq1_upper_bound - sq1_start;
        const auto r_shift2 = chunk2_end - 1 - sq2_start;
        out_key[sq1_start + r_shift1 + r_shift2] = chunk2_r_item;
        out_val[sq1_start + r_shift1 + r_shift2] = in_val[chunk2_end - 1];
      }

      for (auto idx = chunk2_start + 1; idx < chunk2_end - 1; ++idx) {
        const auto inter_item_2 = in_key[idx];
        l_sq1_upper_bound = upper_bound(
            in_key, l_sq1_upper_bound, r_sq1_upper_bound, inter_item_2, comp_t);
        const auto shift1 = l_sq1_upper_bound - sq1_start;
        const auto shift2 = idx - sq2_start;
        out_key[sq1_start + shift1 + shift2] = inter_item_2;
        out_val[sq1_start + shift1 + shift2] = in_val[idx];
      }
    }
  }
}

template <int vec_size, typename KeyType, typename ValueType>
void vec_copy_kernel_impl(
    KeyType* key,
    KeyType* tmp_key_data,
    ValueType* val,
    ValueType* tmp_val_data,
    const size_t sort_sz) {
  auto& q = dpcppGetCurrentQueue();
  using key_vec_t = at::native::Memory::aligned_vector_loop<KeyType, vec_size>;
  using val_vec_t =
      at::native::Memory::aligned_vector_loop<ValueType, vec_size>;
  key_vec_t* key_vec_ptr = reinterpret_cast<key_vec_t*>(key);
  key_vec_t* tmp_key_vec_ptr = reinterpret_cast<key_vec_t*>(tmp_key_data);
  val_vec_t* val_vec_ptr = reinterpret_cast<val_vec_t*>(val);
  val_vec_t* tmp_val_vec_ptr = reinterpret_cast<val_vec_t*>(tmp_val_data);
  auto num_work_item = CeilDiv(sort_sz, (size_t)vec_size);
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item) {
      auto item_id = item.get_linear_id();
      int remaining = sort_sz - item_id * vec_size;
      if (remaining < vec_size) {
        for (int index = 0; index < remaining; index++) {
          auto offset = item_id * vec_size + index;
          key[offset] = tmp_key_data[offset];
          val[offset] = tmp_val_data[offset];
        }
      } else {
#pragma unroll
        for (int index = 0; index < vec_size; index++) {
          key_vec_ptr[item_id][index] = tmp_key_vec_ptr[item_id][index];
          val_vec_ptr[item_id][index] = tmp_val_vec_ptr[item_id][index];
        }
      }
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(num_work_item), kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}

template <typename KeyType, typename ValueType>
void copy_to_dst(
    KeyType* key,
    KeyType* tmp_key_data,
    ValueType* val,
    ValueType* tmp_val_data,
    const size_t sort_sz) {
  int vec_size_key = at::native::Memory::can_vectorize_up_to_loop<KeyType>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(key));
  auto vec_size_val = at::native::Memory::can_vectorize_up_to_loop<ValueType>(
      getDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(val));
  auto vec_size = std::min(vec_size_key, vec_size_val);

#define VEC_COPY_KERNEL_IMPL(vec_size)                  \
  {                                                     \
    vec_copy_kernel_impl<vec_size, KeyType, ValueType>( \
        key, tmp_key_data, val, tmp_val_data, sort_sz); \
  }

  switch (vec_size) {
    case 8: {
      VEC_COPY_KERNEL_IMPL(8);
      break;
    }
    case 4: {
      VEC_COPY_KERNEL_IMPL(4);
      break;
    }
    case 2: {
      VEC_COPY_KERNEL_IMPL(2);
      break;
    }
    case 1: {
      VEC_COPY_KERNEL_IMPL(1);
      break;
    }
    default:
      VEC_COPY_KERNEL_IMPL(1);
  }
#undef VEC_COPY_KERNEL_IMPL
}

// merge sort: only for 1d (single batch) tensor sort
template <typename KeyType, typename ValueType, typename CompFunc>
void merge_sort(
    KeyType* key,
    ValueType* val,
    const size_t sort_sz,
    const CompFunc comp_t) {
  RECORD_FUNCTION("merge_sort", {});
  const size_t leaf = 4;
  const size_t optimal_chunk = 4;

  const size_t leaf_step = ((sort_sz - 1) / leaf) + 1;
  auto& q = dpcppGetCurrentQueue();
  // 1, leaf sort
  auto cgf_1 = DPCPP_Q_CGF(h) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item) {
      leaf_sort<KeyType, ValueType>(item, key, val, leaf, sort_sz, comp_t);
    };
    h.parallel_for(sycl::range<1>(leaf_step), kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf_1);

  auto key_options = map_options<KeyType>();
  auto val_options = map_options<ValueType>();
  Tensor tmp_key = at::empty({sort_sz}, key_options);
  Tensor tmp_val = at::empty({sort_sz}, val_options);
  auto tmp_key_data = tmp_key.data_ptr<KeyType>();
  auto tmp_val_data = tmp_val.data_ptr<ValueType>();

  bool data_in_tmp = false;

  size_t sorted = leaf;
  size_t chunk = std::min(leaf, optimal_chunk);

  while (sorted < sort_sz) {
    size_t sorted_pair = 2 * sorted;
    size_t chunk_num_per_sorted = sorted / chunk;
    size_t full_pairs = sort_sz / sorted_pair;
    size_t incomplete_pair = sort_sz - sorted_pair * full_pairs;
    size_t first_block_in_incomplete_pair =
        incomplete_pair > sorted ? sorted : incomplete_pair;
    size_t incomplete_last_chunk = first_block_in_incomplete_pair % chunk != 0;
    size_t incomplete_pair_steps =
        first_block_in_incomplete_pair / chunk + incomplete_last_chunk;
    size_t full_pair_steps = full_pairs * chunk_num_per_sorted;
    size_t steps = full_pair_steps + incomplete_pair_steps;

    auto cgf_2 = DPCPP_Q_CGF(h) {
      auto kfn = DPCPP_Q_KFN(sycl::item<1> item) {
        const size_t idx = item.get_linear_id();
        const size_t sq1_start =
            std::min(sorted_pair * ((idx * chunk) / sorted), sort_sz);
        const size_t sq1_end = std::min(sq1_start + sorted, sort_sz);
        const size_t sq2_start = sq1_end;
        const size_t sq2_end = std::min(sq2_start + sorted, sort_sz);

        const size_t offset_in_sq = chunk * (idx % chunk_num_per_sorted);

        if (!data_in_tmp) {
          merge(
              offset_in_sq,
              key,
              val,
              tmp_key_data,
              tmp_val_data,
              sq1_start,
              sq1_end,
              sq2_start,
              sq2_end,
              chunk,
              comp_t);
        } else {
          merge(
              offset_in_sq,
              tmp_key_data,
              tmp_val_data,
              key,
              val,
              sq1_start,
              sq1_end,
              sq2_start,
              sq2_end,
              chunk,
              comp_t);
        }
      };
      h.parallel_for(sycl::range<1>(steps), kfn);
    };
    DPCPP_Q_SUBMIT(q, cgf_2);

    data_in_tmp = !data_in_tmp;
    sorted = sorted_pair;
    if (chunk < optimal_chunk)
      chunk *= 2;
  }
  if (data_in_tmp) {
    copy_to_dst<KeyType, ValueType>(
        key, tmp_key_data, val, tmp_val_data, sort_sz);
  }
}

// xpu::pstl::sort for non-batched tensor sort case.
// we have two sort API: one for user defined compare function; one for
// descending/ascending
//
// sort (out_key, out_val, sort_sz, comp_t)
// out_key: result of sort, it is a copy of tensor to be sorted
// out_val: indices of sort, it is initialized by [0, 1, 2, ...]
// sort_sz: element number to be sorted
// comp_t: compare function defined by user

// sort (in_key, out_key, out_val, sort_sz, descending)
// in_key: input tensor to be sorted
// out_key: result of sort, it is a copy of tensor to be sorted
// out_val: indices of sort, it is initialized by [0, 1, 2, ...]
// sort_sz: element number to be sorted
// descending: True for descending, False for ascending.
template <typename KeyType, typename ValueType, typename CompFunc>
void sort(
    KeyType* out_key,
    ValueType* out_val,
    const int64_t sort_sz,
    const CompFunc comp_t) {
  RECORD_FUNCTION("pstl::sort", {});
  merge_sort<KeyType, ValueType>(out_key, out_val, sort_sz, comp_t);
}

template <typename KeyType, typename ValueType>
void sort(
    const KeyType* in_key,
    KeyType* out_key,
    ValueType* out_val,
    const int64_t sort_sz,
    bool descending) {
  RECORD_FUNCTION("pstl::sort", {});
  int stride = 1;
  using offset_t = uint32_t;
  SegmentedGroupRadixSortDesc desc(1, sort_sz, stride, descending, true);

  if (desc.valid()) {
    if (!desc.need_temp()) {
      segmented_group_radix_sort_kernel<KeyType, int64_t, uint16_t, true>(
          desc,
          in_key,
          (KeyType*)out_key,
          nullptr,
          (int64_t*)out_val,
          [=](offset_t slice) -> offset_t { return slice * sort_sz; });
    } else {
      auto key_options = map_options<KeyType>();
      auto val_options = map_options<ValueType>();
      Tensor tmp_key = at::empty({sort_sz}, key_options);
      Tensor tmp_val = at::empty({sort_sz}, val_options);

      segmented_group_radix_sort_kernel<KeyType, int64_t, uint16_t, true>(
          desc,
          in_key,
          (KeyType*)out_key,
          nullptr,
          (int64_t*)out_val,
          [=](offset_t slice) -> offset_t { return slice * sort_sz; },
          (KeyType*)tmp_key.data_ptr(),
          (int64_t*)tmp_val.data_ptr());
    }
  } else {
    if (descending) {
      merge_sort<KeyType, ValueType>(
          out_key, out_val, sort_sz, [](KeyType a, KeyType b) {
            return Numerics<KeyType>::gt(a, b);
          });
    } else {
      merge_sort<KeyType, ValueType>(
          out_key, out_val, sort_sz, [](KeyType a, KeyType b) {
            return Numerics<KeyType>::lt(a, b);
          });
    }
  }
}

} // namespace pstl
} // namespace xpu
