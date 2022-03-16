#pragma once

#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <cmath>

#include "Algorithm.h"
#include "Numerics.h"
#include "Pointwise.h"
#include "SimpleReduce.h"
using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
DPCPP_DEVICE struct AddOp {
  scalar_t operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return lhs + rhs;
  }
};

template <typename scalar_t>
DPCPP_DEVICE struct MulOp {
  scalar_t operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return lhs * rhs;
  }
};

DPCPP_DEVICE struct LogicalAll {
  unsigned char operator()(unsigned char const x, unsigned char const y) const {
    return (x && y);
  }
};

DPCPP_DEVICE struct LogicalAny {
  unsigned char operator()(unsigned char const x, unsigned char const y) const {
    return (x || y);
  }
};

template <typename T>
DPCPP_DEVICE struct ReduceAdd {
  T operator()(const T a, const T b) const {
    return Numerics<T>::add(a, b);
  }
};

template <typename T>
DPCPP_DEVICE struct ReduceMin {
  T operator()(T a, T b) const {
    return (Numerics<T>::lt(a, b) || Numerics<T>::isnan(a)) ? a : b;
  }
};

template <typename T>
DPCPP_DEVICE struct ReduceMax {
  T operator()(T a, T b) const {
    return (Numerics<T>::gt(a, b) || Numerics<T>::isnan(a)) ? a : b;
  }
};

template <typename K, typename Index, class BinaryFunction>
DPCPP_DEVICE void kernelTransformReduceInnermostDimIndex(
    at::Tensor& tgt1,
    at::Tensor& tgt2,
    at::Tensor& src,
    std::pair<K, Index> init,
    BinaryFunction binary_op) {
  auto rdim = tgt1.dim() - 1;
  const auto N = src.size(rdim);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto max_wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  auto wgroup_size = std::min(max_wgroup_size, N);
  auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  tgt1.copy_(src);
  K* tgt1_ptr = tgt1.data_ptr<K>();
  Index* tgt2_ptr = tgt2.data_ptr<Index>();

  const auto num_of_elems = src.numel();
  const auto num_of_reduce_line = src.numel() / N;
  const auto num_dim = (src.dim() == 0) ? 1 : src.dim();

  auto size_rdim = src.size(rdim);

  using max_t = std::pair<K, Index>;

  // initialize tgt2
  auto cgf_0 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      tgt2_ptr[item_id] = item_id % size_rdim;
    };

    __cgh.parallel_for(DPCPP::range<1>(src.numel()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_0);

  if (size_rdim <= max_wgroup_size) {
    wgroup_size = std::min(max_wgroup_size, num_of_reduce_line);
    ngroups = (num_of_reduce_line + wgroup_size - 1) / wgroup_size;
    auto cgf = DPCPP_Q_CGF(__cgh) {
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto local_id = item_id.get_local_linear_id();
        auto global_id = item_id.get_global_linear_id();

        auto head_index = global_id * size_rdim;
        if (head_index < num_of_elems) {
          for (auto i = 1; i < size_rdim; i++) {
            auto index = head_index + i;
            auto head_elem = max_t(tgt1_ptr[head_index], tgt2_ptr[head_index]);
            auto cur_elem = max_t(tgt1_ptr[index], tgt2_ptr[index]);
            auto result_elem = binary_op(head_elem, cur_elem);
            tgt1_ptr[head_index] = result_elem.first;
            tgt2_ptr[head_index] = result_elem.second;
          }
        }
      };

      __cgh.parallel_for(
          DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    return;
  }

  auto remained = size_rdim;
  for (auto ng = ngroups; remained > 1;
       ng = (ng + wgroup_size - 1) / wgroup_size) {
    auto total_ngroups = ng * num_of_reduce_line;
    auto cgf = DPCPP_Q_CGF(__cgh) {
      dpcpp_local_acc_t<max_t> local_max_buf(wgroup_size, __cgh);
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto local_id = item_id.get_local_linear_id();
        auto global_id = item_id.get_global_linear_id();
        auto group_id = item_id.get_group_linear_id();
        auto group_size = item_id.get_local_range().size();
        auto sub_group_id = group_id % ng;
        auto i = group_id / ng;
        auto sub_global_id = global_id % (ng * group_size);

        auto index = i * size_rdim + sub_global_id;
        // init local shared memory of input
        if (sub_global_id < N) {
          local_max_buf[local_id] = max_t(tgt1_ptr[index], tgt2_ptr[index]);
        }

        simple_reduce(item_id, local_max_buf, binary_op);

        auto target_idx = i * size_rdim + sub_group_id;
        if (local_id == 0) {
          tgt1_ptr[target_idx] = local_max_buf[local_id].first;
          tgt2_ptr[target_idx] = local_max_buf[local_id].second;
        }
      };

      __cgh.parallel_for(
          DPCPP::nd_range<1>(total_ngroups * wgroup_size, wgroup_size), kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    remained = ng;
  }
}

template <typename K, typename Index, class BinaryFunction>
DPCPP_DEVICE void kernelTransformReduceOuterDimIndex(
    at::Tensor& tgt1,
    at::Tensor& tgt2,
    const at::Tensor& src,
    int64_t rdim,
    std::pair<K, Index> init,
    BinaryFunction binary_op) {
  const auto N = src.size(rdim);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto max_wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  auto wgroup_size = std::min(max_wgroup_size, N);
  auto ngroups = (N + wgroup_size - 1) / wgroup_size;

  tgt1.copy_(src);
  K* tgt1_ptr = tgt1.data_ptr<K>();
  Index* tgt2_ptr = tgt2.data_ptr<Index>();

  const auto num_of_elems = src.numel();
  const auto num_of_reduce_line = src.numel() / N;
  const auto num_dim = (src.dim() == 0) ? 1 : src.dim();

  auto stride_rdim = src.stride(rdim);
  auto size_rdim = src.size(rdim);

  using max_t = std::pair<K, Index>;

  // initialize tgt2
  auto cgf_0 = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      tgt2_ptr[item_id] = item_id / stride_rdim % size_rdim;
    };

    __cgh.parallel_for(DPCPP::range<1>(src.numel()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_0);

  if (size_rdim <= max_wgroup_size) {
    wgroup_size = std::min(max_wgroup_size, num_of_reduce_line);
    ngroups = (num_of_reduce_line + wgroup_size - 1) / wgroup_size;
    auto cgf = DPCPP_Q_CGF(__cgh) {
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto local_id = item_id.get_local_linear_id();
        auto global_id = item_id.get_global_linear_id();

        auto head_index = global_id / stride_rdim * stride_rdim * size_rdim +
            global_id % stride_rdim;
        if (head_index < num_of_elems) {
          for (auto i = 1; i < size_rdim; i++) {
            auto index = head_index + i * stride_rdim;
            auto head_elem = max_t(tgt1_ptr[head_index], tgt2_ptr[head_index]);
            auto cur_elem = max_t(tgt1_ptr[index], tgt2_ptr[index]);
            auto result_elem = binary_op(head_elem, cur_elem);
            tgt1_ptr[head_index] = result_elem.first;
            tgt2_ptr[head_index] = result_elem.second;
          }
        }
      };

      __cgh.parallel_for(
          DPCPP::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    return;
  }

  auto remained = size_rdim;
  for (auto ng = ngroups; remained > 1;
       ng = (ng + wgroup_size - 1) / wgroup_size) {
    auto total_ngroups = ng * num_of_reduce_line;
    auto cgf = DPCPP_Q_CGF(__cgh) {
      dpcpp_local_acc_t<max_t> local_max_buf(wgroup_size, __cgh);
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto local_id = item_id.get_local_linear_id();
        auto global_id = item_id.get_global_linear_id();
        auto group_id = item_id.get_group_linear_id();
        auto group_size = item_id.get_local_range().size();
        auto sub_group_id = group_id % ng;
        auto i = group_id / ng;
        auto sub_global_id = global_id % (ng * group_size);

        auto index = i / stride_rdim * stride_rdim * size_rdim +
            i % stride_rdim + sub_global_id * stride_rdim;
        // init local shared memory of input
        if (sub_global_id < N) {
          local_max_buf[local_id] = max_t(tgt1_ptr[index], tgt2_ptr[index]);
        }

        simple_reduce(item_id, local_max_buf, binary_op);

        auto target_idx = i / stride_rdim * stride_rdim * size_rdim +
            i % stride_rdim + sub_group_id * stride_rdim;
        if (local_id == 0) {
          tgt1_ptr[target_idx] = local_max_buf[local_id].first;
          tgt2_ptr[target_idx] = local_max_buf[local_id].second;
        }
      };

      __cgh.parallel_for(
          DPCPP::nd_range<1>(total_ngroups * wgroup_size, wgroup_size), kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    remained = ng;
  }
}

template <
    typename ScalarTypeK,
    typename ScalarTypeIndex,
    typename BinaryFunction>
DPCPP_HOST void transformReduceOuterDimIndex(
    at::Tensor& tgt1,
    at::Tensor& tgt2,
    const at::Tensor& src,
    int64_t rdim,
    const std::pair<ScalarTypeK, ScalarTypeIndex>& init,
    BinaryFunction binary_op) {
  kernelTransformReduceOuterDimIndex(tgt1, tgt2, src, rdim, init, binary_op);
}

template <
    typename ScalarTypeK,
    typename ScalarTypeIndex,
    typename BinaryFunction>
DPCPP_HOST void transformReduceInnermostDimIndex(
    at::Tensor& tgt1,
    at::Tensor& tgt2,
    at::Tensor& src,
    const std::pair<ScalarTypeK, ScalarTypeIndex>& init,
    BinaryFunction binary_op) {
  kernelTransformReduceInnermostDimIndex(tgt1, tgt2, src, init, binary_op);
}

template <
    typename ScalarTypeK,
    typename ScalarTypeIndex,
    typename BinaryFunction>
DPCPP_HOST void reduceDimIndex(
    at::Tensor& tgt1_,
    at::Tensor& tgt2_,
    const at::Tensor& src_,
    int64_t dimension,
    int keepdim,
    const std::pair<ScalarTypeK, ScalarTypeIndex> init,
    BinaryFunction binary_op) {
  TORCH_CHECK(
      dimension >= 0 && dimension < src_.dim(), "dimension out of range");

  // Unsqueeze tgt1_/tgt_2 if necessary so that their contiguity traits
  // are preserved if they are the same size as the correct reduction output.
  int src_dims = src_.dim();
  TensorImpl_preserveReduceDimSemantics(
      TensorImpl_Unwrap(tgt1_), src_dims, dimension, keepdim);
  TensorImpl_preserveReduceDimSemantics(
      TensorImpl_Unwrap(tgt2_), src_dims, dimension, keepdim);

  auto src = src_.contiguous();
  auto tgt1 = tgt1_.contiguous();
  auto tgt2 = tgt2_.contiguous();
  tgt1.resize_as_(src);
  tgt2.resize_as_(src);

  if (dimension == ((src_.dim() == 0 ? 1 : src_.dim()) - 1)) {
    transformReduceInnermostDimIndex(tgt1, tgt2, src, init, binary_op);
  } else {
    transformReduceOuterDimIndex(tgt1, tgt2, src, dimension, init, binary_op);
  }

  std::vector<int64_t> dim = src_.sizes().vec();
  dim[dimension] = 1;
  tgt1_ = tgt1_.narrow(dimension, 0, 1).contiguous();
  tgt2_ = tgt2_.narrow(dimension, 0, 1).contiguous();

  if (!keepdim) {
    TensorImpl_squeeze1d(
        TensorImpl_Unwrap(tgt1_), TensorImpl_Unwrap(tgt1_), dimension);
    TensorImpl_squeeze1d(
        TensorImpl_Unwrap(tgt2_), TensorImpl_Unwrap(tgt2_), dimension);
  }
}

template <typename T, typename Index>
struct MaxValuePair {
  std::pair<T, Index> operator()(
      const std::pair<T, Index> a,
      const std::pair<T, Index> b) const {
    return (Numerics<T>::ge(a.first, b.first) || Numerics<T>::isnan(a.first))
        ? a
        : b;
  }
};

template <typename T, typename Index>
struct MinValuePair {
  std::pair<T, Index> operator()(
      const std::pair<T, Index> a,
      const std::pair<T, Index> b) const {
    return (Numerics<T>::le(a.first, b.first) || Numerics<T>::isnan(a.first))
        ? a
        : b;
  }
};

} // namespace AtenIpexTypeXPU
} // namespace at
