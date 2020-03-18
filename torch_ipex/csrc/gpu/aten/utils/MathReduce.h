#ifndef REDUCEDIMINDEX_H
#define REDUCEDIMINDEX_H

#include <ATen/ATen.h>

#include <core/DPCPP.h>
#include <core/Memory.h>
#include <core/DPCPPUtils.h>
#include <core/TensorImplUtils.h>

#include <utils/Numerics.h>
#include <utils/Algorithm.h>


using namespace at::native;
using namespace at::dpcpp;

template <typename scalar_t>
DP_DEVICE struct AddOp {
  scalar_t operator() (const scalar_t& lhs, const scalar_t& rhs) const {
    return lhs + rhs;
  }
};

template <typename scalar_t>
DP_DEVICE struct MulOp {
  scalar_t operator() (const scalar_t& lhs, const scalar_t& rhs) const {
    return lhs * rhs;
  }
};

DP_DEVICE struct LogicalAll {
  unsigned char operator()(unsigned char const x, unsigned char const y) const {
    return (x && y);
  }
};

DP_DEVICE struct LogicalAny {
  unsigned char operator()(unsigned char const x, unsigned char const y) const {
    return (x || y);
  }
};

template <typename T>
DP_DEVICE struct ReduceAdd {
  T operator()(const T a, const T b) const {
    return Numerics<T>::add(a, b);
  }
};

template <typename T>
DP_DEVICE struct ReduceMin {
  T operator()(T a, T b) const {
    return (Numerics<T>::lt(a, b) || Numerics<T>::isnan(a)) ? a : b;
  }
};

template <typename T>
DP_DEVICE struct ReduceMax {
  T operator()(T a, T b) const {
    return (Numerics<T>::gt(a, b) || Numerics<T>::isnan(a)) ? a : b;
  }
};

DP_DEF_K1(reduceInnermostDimIndex);

DP_DEF_K1(reduceOuterDimIndex);

template <typename K,
          typename Index,
          class BinaryFunction>
DP_DEVICE void
kernelTransformReduceInnermostDimIndex(at::Tensor & tgt1,
                                       at::Tensor & tgt2,
                                       at::Tensor & src,
                                       std::pair<K, Index> init,
                                       BinaryFunction binary_op) {

  auto queue         = dpcppGetCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(queue);
  auto totalElements = src.numel();
  auto num_groups    = CeilDiv(totalElements, group_size);
  auto total_items   = num_groups * group_size;

  auto tgt1_data     = tgt1.data<K>();
  auto tgt2_data     = tgt2.data<Index>();
  auto tgt1_size     = (tgt1.numel()) * (tgt1.itemsize());
  auto tgt2_size     = (tgt2.numel()) * (tgt2.itemsize());
  auto src_data      = src.data<K>();
  auto src_size      = totalElements * (src.itemsize());
  auto dim           = tgt1.dim() - 1;

  int64_t n = src.size(dim);
  int64_t stride = src.stride(dim);
  int64_t batch = totalElements / (n * stride);

  auto cgf = DP_Q_CGF(cgh) {
    auto src_acc  = DPCPPAccessor<dp_r_mode>(cgh, src_data, src_size);
    auto tgt1_acc = DPCPPAccessor<dp_w_mode>(cgh, tgt1_data, tgt1_size);
    auto tgt2_acc = DPCPPAccessor<dp_w_mode>(cgh, tgt2_data, tgt2_size);

    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto src_ptr  = src_acc.template get_pointer<K>();
      auto tgt1_ptr = tgt1_acc.template get_pointer<K>();
      auto tgt2_ptr = tgt2_acc.template get_pointer<Index>();

      for (int64_t linearIndex = item.get_global_id(0);
           linearIndex < totalElements; linearIndex += item.get_global_range()[0]) {
        int64_t base_start = linearIndex % (batch * stride);
        int64_t start = linearIndex % batch * n;

        std::pair<K, Index> acc = init;
        for (int64_t j = 0; j < n; ++j) {
          //
          // The explicit typecast looks weired but is necessory to solve the build error
          // "candidate function not viable: no known conversion from '__global long' to
          // '__global long &&' for 1st argument."
          //
          K data = src_ptr[start + j];
          Index idx = j /* + TH_INDEX_BASE */;
          acc = binary_op (acc, std::make_pair<K, Index>((K)data, (Index)idx));
      }
        tgt1_ptr[base_start] = acc.first;
        tgt2_ptr[base_start] = acc.second;
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(reduceInnermostDimIndex, K, Index, BinaryFunction)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };

  // submit to DPCPP queue
  DP_Q_ASYNC_SUBMIT(queue, cgf);
}

template <typename K,
          typename Index,
          class BinaryFunction>
DP_DEVICE void
kernelTransformReduceOuterDimIndex(at::Tensor & tgt1,
                                   at::Tensor & tgt2,
                                   at::Tensor & src,
                                   int64_t rdim,
                                   std::pair<K, Index> init,
                                   BinaryFunction binary_op) {

  auto queue         = dpcppGetCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(queue);
  auto totalElements = src.numel();
  auto num_groups    = CeilDiv(totalElements, group_size);
  auto total_items   = num_groups * group_size;

  auto tgt1_data     = tgt1.data<K>();
  auto tgt2_data     = tgt2.data<Index>();
  auto tgt1_size     = (tgt1.numel()) * (tgt1.itemsize());
  auto tgt2_size     = (tgt2.numel()) * (tgt2.itemsize());
  auto src_data      = src.data<K>();
  auto src_size      = totalElements * (src.itemsize());

  int64_t n = src.size(rdim);
  int64_t stride = src.stride(rdim);
  int64_t batch = totalElements / (n * stride);

  auto cgf = DP_Q_CGF(cgh) {
    auto src_acc  = DPCPPAccessor<dp_r_mode>(cgh, src_data, src_size);
    auto tgt1_acc = DPCPPAccessor<dp_w_mode>(cgh, tgt1_data, tgt1_size);
    auto tgt2_acc = DPCPPAccessor<dp_w_mode>(cgh, tgt2_data, tgt2_size);

    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto src_ptr  = src_acc.template get_pointer<K>();
      auto tgt1_ptr = tgt1_acc.template get_pointer<K>();
      auto tgt2_ptr = tgt2_acc.template get_pointer<Index>();

      for (int64_t linearIndex = item.get_global_id(0);
           linearIndex < totalElements; linearIndex += item.get_global_range()[0]) {
        int64_t base_start = linearIndex % (batch * stride);
        int64_t start = (base_start / stride) * n * stride + base_start % stride;

        std::pair<K, Index> acc = init;
        for (int64_t j = 0; j < n; ++j) {
          //
          // The explicit typecast looks weired but is necessory to solve the build error
          // "candidate function not viable: no known conversion from '__global long' to
          // '__global long &&' for 1st argument."
          //
          K data = src_ptr[start + j * stride];
          Index idx = j /* + TH_INDEX_BASE */;
          acc = binary_op (acc, std::make_pair<K, Index>((K)data, (Index)idx));
          tgt1_ptr[base_start] = acc.first;
          tgt2_ptr[base_start] = acc.second;
        }
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(reduceOuterDimIndex, K, Index, BinaryFunction)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };

  // submit to DPCPP queue
  DP_Q_ASYNC_SUBMIT(queue, cgf);
};

template <typename ScalarTypeK,
          typename ScalarTypeIndex,
          typename BinaryFunction>
DP_HOST void
transformReduceOuterDimIndex(at::Tensor & tgt1,
                             at::Tensor & tgt2,
                             at::Tensor & src,
                             int64_t rdim,
                             const std::pair<ScalarTypeK, ScalarTypeIndex>& init,
                             BinaryFunction binary_op) {
  kernelTransformReduceOuterDimIndex(tgt1, tgt2, src, rdim, init, binary_op);
}

template <typename ScalarTypeK,
          typename ScalarTypeIndex,
          typename BinaryFunction>
DP_HOST void
transformReduceInnermostDimIndex(at::Tensor & tgt1,
                                 at::Tensor & tgt2,
                                 at::Tensor & src,
                                 const std::pair<ScalarTypeK, ScalarTypeIndex>& init,
                                 BinaryFunction binary_op) {
  kernelTransformReduceInnermostDimIndex(tgt1, tgt2, src, init, binary_op);
}

template <typename ScalarTypeK,
          typename ScalarTypeIndex,
          typename BinaryFunction>
DP_HOST void
reduceDimIndex(at::Tensor & tgt1_,
               at::Tensor & tgt2_,
               const at::Tensor & src_,
               int64_t dimension,
               int keepdim,
               const std::pair<ScalarTypeK, ScalarTypeIndex> init,
               BinaryFunction binary_op) {
  TORCH_CHECK(dimension >= 0 && dimension < src_.dim(), "dimension out of range");

  // Unsqueeze tgt1_/tgt_2 if necessary so that their contiguity traits
  // are preserved if they are the same size as the correct reduction output.
  int src_dims = src_.dim();
  TensorImpl_preserveReduceDimSemantics(
      TensorImpl_Unwrap(tgt1_), src_dims, dimension, keepdim);
  TensorImpl_preserveReduceDimSemantics(
      TensorImpl_Unwrap(tgt2_), src_dims, dimension, keepdim);


  std::vector<int64_t> dim;
  for (int i = 0; i < src_.dim(); i++)
    dim.push_back(src_.sizes()[i]);
  dim[dimension] = 1;
  tgt1_.resize_(dim);
  tgt2_.resize_(dim);

  auto tgt1 = tgt1_.contiguous();
  auto tgt2 = tgt2_.contiguous();
  auto src = src_.contiguous();

  if (dimension == ((src.dim() == 0 ? 1 : src.dim()) - 1)) {
    transformReduceInnermostDimIndex(tgt1, tgt2, src, init, binary_op);
  } else {
    transformReduceOuterDimIndex(tgt1, tgt2, src, dimension, init, binary_op);
  }

  if (!keepdim) {
    TensorImpl_squeeze1d(
        TensorImpl_Unwrap(tgt1_), TensorImpl_Unwrap(tgt1_), dimension);
    TensorImpl_squeeze1d(
        TensorImpl_Unwrap(tgt2_), TensorImpl_Unwrap(tgt2_), dimension);
  }
}

template <typename T, typename Index>
struct MaxValuePair {
  std::pair<T, Index> operator()(const std::pair<T, Index> a,
                                 const std::pair<T, Index> b) const{
    return (Numerics<T>::ge(a.first, b.first) ||
            Numerics<T>::isnan(a.first)) ? a : b;
  }
};

template <typename T, typename Index>
struct MinValuePair {
  std::pair<T, Index> operator()(const std::pair<T, Index> a,
                                 const std::pair<T, Index> b) const {
    return (Numerics<T>::le(a.first, b.first) ||
            Numerics<T>::isnan(a.first)) ? a : b;
  }
};

#endif
