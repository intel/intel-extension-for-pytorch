#pragma once

#include <utils/DPCPP.h>
#include "Atomics.h"
#include "SimpleReduce.h"

#ifdef USE_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

namespace sycl {
template <typename T>
struct sycl::is_device_copyable<oneapi::dpl::zip_iterator<T*, T*>>
    : std::true_type {};
} // namespace sycl
#endif

namespace at {
namespace AtenIpexTypeXPU {

template <class InputIt, class OutputIt, class T>
DPCPP_DEVICE static inline OutputIt exclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  // 1. do exclusive_scan on each workgroups
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    DPCPP::accessor<T, 1, dpcpp_rw_mode, DPCPP::access::target::local>
        local_scan(wgroup_size, __cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto local_id = item_id.get_local_linear_id();
      auto global_id = item_id.get_global_linear_id();
      auto group_id = item_id.get_group_linear_id();
      auto group_size = item_id.get_local_range().size();

      if (first + global_id < last) {
        local_scan[local_id] = first[global_id];
      } else {
        local_scan[local_id] = 0;
      }

      up_sweep(item_id, local_scan);
      down_sweep(item_id, local_scan, init);

      if (first + global_id < last)
        d_first[global_id] = local_scan[local_id];
    };

    __cgh.parallel_for(
        DPCPP::nd_range</*dim=*/1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. reduce among workgroups,
  //     will rewrite this while work group reduction is ready For now,
  //     this is implemented with serial algorithm actually.
  auto cgf_2 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto local_id = item_id.get_local_linear_id();
      auto group_size = item_id.get_local_range().size();

      for (auto i = 1; i <= ngroups; i++) {
        auto global_id = i * wgroup_size + local_id;
        auto prelast_id = i * wgroup_size - 1;
        if (global_id >= N)
          continue;
        d_first[global_id] =
            d_first[global_id] + d_first[prelast_id] + first[prelast_id];
      }
    };
    __cgh.parallel_for(
        DPCPP::nd_range</*dim=*/1>(wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_2);

  return d_first + N;
}

template <class InputIt, class OutputIt, class T>
DPCPP_DEVICE static inline OutputIt inclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  // 1. do exclusive_scan
  OutputIt d_last =
      at::AtenIpexTypeXPU::exclusive_scan(first, last, d_first, init);
  const auto N = std::distance(d_first, d_last);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  // 2. update exclusive_scan to inclusive_scan
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto global_id = item_id.get_global_linear_id();
      d_first[global_id] += first[global_id];
    };

    __cgh.parallel_for(
        DPCPP::nd_range</*dim=*/1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return d_first + N;
}

template <class InputIt, class OutputIt, class UnaryPredicate>
DPCPP_DEVICE static inline OutputIt copy_if(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    UnaryPredicate pred) {
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  Tensor global_mask = at::zeros(
      {N + 1},
      at::TensorOptions().device(kXPU).dtype(kLong).memory_format(
          LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  int64_t* gmask_ptr = (int64_t*)global_mask.data_ptr();
  Tensor target_pos = at::empty(
      {N + 1},
      at::TensorOptions().device(kXPU).dtype(kLong).memory_format(
          LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  int64_t* tpos_ptr = (int64_t*)target_pos.data_ptr();

  // 1. get mask vector (Tensor) for exclusive_scan
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto global_id = item_id.get_global_linear_id();

      if (first + global_id < last) {
        gmask_ptr[global_id] = (int64_t)pred(first[global_id]);
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. calculate positions via exclusive_scan
  at::AtenIpexTypeXPU::exclusive_scan(
      gmask_ptr, gmask_ptr + N + 1, tpos_ptr, (int64_t)0);

  // 3. copy data into destination
  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto global_id = item_id.get_global_linear_id();

      if (first + global_id < last) {
        auto pos = tpos_ptr[global_id];
        if (gmask_ptr[global_id] != 0) {
          d_first[pos] = first[global_id];
        }
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  return d_first + target_pos.max().item<int64_t>();
}

template <class InputIt, class OutputIt, class UnaryOperation>
DPCPP_DEVICE static inline OutputIt transform(
    InputIt first1,
    InputIt last1,
    OutputIt d_first,
    UnaryOperation unary_op) {
  const auto N = std::distance(first1, last1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto global_id = item_id.get_global_linear_id();

      if (first1 + global_id < last1) {
        d_first[global_id] = unary_op(first1[global_id]);
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return d_first + N;
}

template <class InputIt1, class InputIt2, class OutputIt, class BinaryOperation>
DPCPP_DEVICE static inline OutputIt transform(
    InputIt1 first1,
    InputIt1 last1,
    InputIt2 first2,
    OutputIt d_first,
    BinaryOperation binary_op) {
  const auto N = std::distance(first1, last1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto global_id = item_id.get_global_linear_id();

      if (first1 + global_id < last1) {
        d_first[global_id] = binary_op(first1[global_id], first2[global_id]);
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return d_first + N;
}

template <class ForwardIt, class T>
DPCPP_DEVICE static inline void iota(ForwardIt first, ForwardIt last, T value) {
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto global_id = item_id.get_global_linear_id();

      if (first + global_id < last) {
        first[global_id] = value + global_id;
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <class ForwardIt, class BinaryPredicate>
ForwardIt unique(ForwardIt first, ForwardIt last, BinaryPredicate p) {
  const auto N = std::distance(first, last);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  Tensor global_mask = at::ones(
      {N + 1},
      at::TensorOptions().device(kXPU).dtype(kLong).memory_format(
          LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  int64_t* gmask_ptr = (int64_t*)global_mask.data_ptr();
  Tensor target_pos = at::empty(
      {N + 1},
      at::TensorOptions().device(kXPU).dtype(kLong).memory_format(
          LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  int64_t* tpos_ptr = (int64_t*)target_pos.data_ptr();

  // 1. mark duplicated item as 0 in mask vector.
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto global_id = item_id.get_global_linear_id();
      if (global_id > 1 && first + global_id < last) {
        if (p(first[global_id - 1], first[global_id])) {
          gmask_ptr[global_id] = 0;
        }
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  // 2. calculate positions via exclusive_scan
  at::AtenIpexTypeXPU::exclusive_scan(
      gmask_ptr, gmask_ptr + N + 1, tpos_ptr, (int64_t)0);

  // 3. copy data into destination, now its done by linear algorithm, need to
  // rewrite this while global reduce is ready
  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto global_id = item_id.get_global_linear_id();
      if (global_id == 0) {
        for (auto i = 0; first + i < last; i++) {
          auto pos = tpos_ptr[i];
          if (gmask_ptr[i] != 0) {
            *(first + pos) = std::move(*(first + i));
          }
        }
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  return first + target_pos.max().item<int64_t>();
}

} // namespace AtenIpexTypeXPU
} // namespace at
