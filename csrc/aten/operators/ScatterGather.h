#include <core/Memory.h>
#include <core/Stream.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

enum class SCATTER_GATHER_OP : uint8_t { REDUCE_ADD, REDUCE_MULTIPLY };

SCATTER_GATHER_OP get_operator_enum(const c10::string_view reduce) {
  if (reduce == "add") {
    return SCATTER_GATHER_OP::REDUCE_ADD;
  } else if (reduce == "multiply") {
    return SCATTER_GATHER_OP::REDUCE_MULTIPLY;
  } else {
    TORCH_CHECK(false, "reduce argument must be either add or multiply.");
  }
}

template <typename T, typename ReduceStub, typename FillStub>
void scatter_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const T& src,
    const Tensor& out,
    ReduceStub& reduce_stub,
    FillStub& fill_stub,
    const c10::optional<c10::string_view> reduce = nullopt) {
  if (index.numel() == 0)
    return;
  dim = at::maybe_wrap_dim(dim, self.dim());
  auto mut_out = const_cast<Tensor&>(out);

  if (!self.is_same(mut_out)) {
    mut_out.copy_(self);
  }

  if (reduce.has_value()) {
    auto op = get_operator_enum(reduce.value());
    reduce_stub(mut_out, dim, index, src, op);
  } else {
    fill_stub(mut_out, dim, index, src);
  }
}

class ReduceMultiply {
 public:
  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicMul((dpcpp_global_ptr_pt<scalar_t>)self_data, *src_data);
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
 public:
  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicAdd((dpcpp_global_ptr_pt<scalar_t>)self_data, *src_data);
  }
};
static ReduceAdd reduce_add;

class TensorAssign {
 public:
  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

// The kernels are implemented on an opaque,
// self-aligned type of the correct size,
// to avoid redundant kernels for different types
// of the same size.
template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

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

    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
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

    __cgh.parallel_for(sycl::range</*dim=*/1>(totalElements), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename IndexType, typename Real, int Dims>
void THSyclTensor_scatterKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  IndexType group_size = (IndexType)dpcppMaxWorkGroupSize(dev_id);
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = tensor.data;
    auto src_data = src.data;
    auto index_data = index.data;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
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
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_items), sycl::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename IndexType, typename Real, int Dims>
void THSyclTensor_scatterAddKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  IndexType group_size = (IndexType)dpcppMaxWorkGroupSize(dev_id);
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = tensor.data;
    auto src_data = src.data;
    auto index_data = index.data;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
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

        atomicAdd(
            (dpcpp_global_ptr_pt<Real>)&tensor_ptr[tensorOffset],
            src_ptr[srcOffset]);
      }
    };

    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_items), sycl::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename IndexType, typename Real, int Dims>
void THSyclTensor_scatterFillKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<int64_t, IndexType> index,
    Real value,
    const int dim,
    const IndexType totalElements) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  IndexType group_size = (IndexType)dpcppMaxWorkGroupSize(dev_id);
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = tensor.data;
    auto index_data = index.data;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
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
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_items), sycl::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace AtenIpexTypeXPU
} // namespace at
