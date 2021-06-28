#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <core/Memory.h>
#include <core/Stream.h>
#include <core/detail/TensorInfo.h>

#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, typename Real, int Dims>
struct IndexToScatterGatherOffsets {
  static DPCPP_DEVICE void compute(
      IndexType linearId,
      const int dim,
      const TensorInfo<int64_t, IndexType>& index,
      IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t1,
      IndexType* t1Offset,
      const TensorInfo<Real, IndexType>& t2,
      IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static DPCPP_DEVICE void compute(
      IndexType linearId,
      const int dim,
      const TensorInfo<int64_t, IndexType>& index,
      IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t2,
      IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

// Same as above but using a dynamic number of dimensions.
template <typename IndexType, typename Real>
struct IndexToScatterGatherOffsets<IndexType, Real, -1> {
  static DPCPP_DEVICE void compute(
      IndexType linearId,
      const int dim,
      const TensorInfo<int64_t, IndexType>& index,
      IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t1,
      IndexType* t1Offset,
      const TensorInfo<Real, IndexType>& t2,
      IndexType* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static DPCPP_DEVICE void compute(
      IndexType linearId,
      const int dim,
      const TensorInfo<int64_t, IndexType>& index,
      IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t2,
      IndexType* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

// DPCPP_DEF_K1(dpcpp_gather_kernel);
template <typename IndexType, typename Real, int Dims>
class dpcpp_gather_kernel {};
template <typename IndexType, typename Real, int Dims>
void THDPCPPTensor_gatherKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto tensor_data = tensor.data;
    auto src_data = src.data;
    auto index_data = index.data;

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto tensor_ptr = tensor_data;
      auto src_ptr = src_data;
      auto index_ptr = index_data;

      auto linear_idx = item_id.get_id(0);

      IndexType tensorOffset = 0;
      IndexType srcOffset = 0;
      IndexType indexOffset = 0;

      IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(
          linear_idx,
          dim,
          index,
          &indexOffset,
          tensor,
          &tensorOffset,
          src,
          &srcOffset);

      int64_t indexValue = index_ptr[indexOffset];
      if (indexValue >= 0 &&
          static_cast<IndexType>(indexValue) < src.sizes[dim]) {
        srcOffset += indexValue * src.strides[dim];

        tensor_ptr[tensorOffset] = src_ptr[srcOffset];
      }
      //      else
      //        add warning
    };

    __cgh.parallel_for</*DPCPP_K(dpcpp_gather_kernel, Real, IndexType, Dims)*/
                       dpcpp_gather_kernel<Real, IndexType, Dims>>(
        DPCPP::range</*dim=*/1>(totalElements), kfn);
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

DPCPP_DEF_K2(scatterSycl, typename IndexType, typename Real, int Dims);
template <typename IndexType, typename Real, int Dims>
void THSyclTensor_scatterKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  auto& queue = dpcppGetCurrentQueue();
  IndexType group_size = (IndexType)dpcppMaxWorkGroupSize(queue);
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = tensor.data;
    auto src_data = src.data;
    auto index_data = index.data;
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto tensor_ptr = out_data;
      auto src_ptr = src_data;
      auto index_ptr = index_data;
      for (IndexType linearIndex = (IndexType)item.get_global_id(0);
           linearIndex < totalElements;
           linearIndex += (IndexType)item.get_global_range()[0]) {
        IndexType tensorOffset = 0;
        IndexType srcOffset = 0;
        IndexType indexOffset = 0;

        IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(
            linearIndex,
            dim,
            index,
            &indexOffset,
            src,
            &srcOffset,
            tensor,
            &tensorOffset);

        int64_t indexValue = index_ptr[indexOffset];
        // assert(indexValue >= 0 && indexValue < src.sizes[dim]);
        tensorOffset += indexValue * tensor.strides[dim];

        tensor_ptr[tensorOffset] = src_ptr[srcOffset];
      }
    };

    // kick off kernel
    cgh.parallel_for<DPCPP_K(scatterSycl, IndexType, Real, Dims)>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

DPCPP_DEF_K2(scatterAddSycl, typename IndexType, typename Real, int Dims);
template <typename IndexType, typename Real, int Dims>
void THSyclTensor_scatterAddKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  auto& queue = dpcppGetCurrentQueue();
  IndexType group_size = (IndexType)dpcppMaxWorkGroupSize(queue);
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = tensor.data;
    auto src_data = src.data;
    auto index_data = index.data;
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto tensor_ptr = out_data;
      auto src_ptr = src_data;
      auto index_ptr = index_data;

      for (IndexType linearIndex = (IndexType)item.get_global_id(0);
           linearIndex < totalElements;
           linearIndex += (IndexType)item.get_global_range()[0]) {
        IndexType tensorOffset = 0;
        IndexType srcOffset = 0;
        IndexType indexOffset = 0;

        IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(
            linearIndex,
            dim,
            index,
            &indexOffset,
            src,
            &srcOffset,
            tensor,
            &tensorOffset);

        int64_t indexValue = index_ptr[indexOffset];
        // assert(indexValue >= 0 && indexValue < src.sizes[dim]);
        tensorOffset += indexValue * tensor.strides[dim];

        atomicAdd((dpcpp_global_ptr_pt<Real>)&tensor_ptr[tensorOffset], src_ptr[srcOffset]);
      }
    };

    // kick off kernel
    cgh.parallel_for<DPCPP_K(scatterAddSycl, IndexType, Real, Dims)>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

DPCPP_DEF_K2(scatterFillSycl, typename IndexType, typename Real, int Dims);
template <typename IndexType, typename Real, int Dims>
void THSyclTensor_scatterFillKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<int64_t, IndexType> index,
    Real value,
    const int dim,
    const IndexType totalElements) {
  auto& queue = dpcppGetCurrentQueue();
  IndexType group_size = (IndexType)dpcppMaxWorkGroupSize(queue);
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = tensor.data;
    auto index_data = index.data;
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      auto tensor_ptr = out_data;
      auto index_ptr = index_data;
      for (IndexType linearIndex = (IndexType)item.get_global_id(0);
           linearIndex < totalElements;
           linearIndex += (IndexType)item.get_global_range()[0]) {
        IndexType tensorOffset = 0;
        IndexType indexOffset = 0;

        IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(
            linearIndex, dim, index, &indexOffset, tensor, &tensorOffset);

        int64_t indexValue = index_ptr[indexOffset];
        // assert(indexValue >= 0 && indexValue < src.sizes[dim]);
        tensorOffset += indexValue * tensor.strides[dim];

        tensor_ptr[tensorOffset] = value;
      }
    };

    // kick off kernel
    cgh.parallel_for<DPCPP_K(scatterFillSycl, IndexType, Real, Dims)>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(total_items), DPCPP::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}
