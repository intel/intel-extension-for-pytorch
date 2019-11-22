#include <c10/dpcpp/SYCL.h>
#include <THDP/THSYCLTensor.hpp>
#include <THDP/THSYCLTensorCopy.hpp>
#include <THDP/THSYCLTensorTypeUtils.h>
#include <THDP/THSYCLDeviceUtils.h>
#include <THDP/THDPAtomics.h>

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, typename Real, int Dims>
struct IndexToScatterGatherOffsets {
  static DP_DEVICE void compute(
          IndexType linearId, const int dim,
          const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
          const TensorInfo<Real, IndexType>& t1, IndexType* t1Offset,
          const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
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

  static DP_DEVICE void compute(
          IndexType linearId, const int dim,
          const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
          const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
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
  static DP_DEVICE void compute(
      IndexType linearId, const int dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
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

  static DP_DEVICE void compute(
      IndexType linearId, const int dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
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

//DP_DEF_K1(sycl_gather_kernel);
template <typename IndexType, typename Real, int Dims> class sycl_gather_kernel{};
template <typename IndexType, typename Real, int Dims>
void THSYCLTensor_gatherKernel(
        TensorInfo<Real, IndexType> tensor,
        TensorInfo<Real, IndexType> src,
        TensorInfo<int64_t, IndexType> index,
        const int dim,
        const IndexType totalElements) {

  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();

  using out_accessor_t = c10::sycl::SYCLAccessor<dp_discard_w_mode>;
  using in_accessor_t = c10::sycl::SYCLAccessor<dp_r_mode>;

  auto cgf = DP_Q_CGF(__cgh) {
    out_accessor_t tensor_acc = out_accessor_t (__cgh, tensor.data);
    in_accessor_t src_acc = in_accessor_t (__cgh, src.data);
    in_accessor_t index_acc = in_accessor_t (__cgh, index.data);

    auto kfn = DP_Q_KFN(DP::item<1> item_id)  {
      auto tensor_data = tensor_acc.template get_pointer<Real>();
      auto src_data = src_acc.template get_pointer<Real>();
      auto index_data = index_acc.template get_pointer<int64_t>();

      auto linear_idx = item_id.get_id(0);

      IndexType tensorOffset = 0;
      IndexType srcOffset = 0;
      IndexType indexOffset = 0;

      IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linear_idx, dim,
                                                                  index, &indexOffset,
                                                                  tensor, &tensorOffset,
                                                                  src, &srcOffset);

      int64_t indexValue = index_data[indexOffset];
      if (indexValue >= 0 && static_cast<IndexType>(indexValue) < src.sizes[dim]) {
        srcOffset += indexValue * src.strides[dim];

        tensor_data[tensorOffset] = src_data[srcOffset];
      }
//      else
//        add warning

    };

    __cgh.parallel_for</*DP_K(sycl_gather_kernel, Real, IndexType, Dims)*/
            sycl_gather_kernel<Real, IndexType, Dims>>(
            DP::range</*dim=*/1>(totalElements), kfn);
  };
  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);

}

DP_DEF_K2(scatterSycl, typename IndexType, typename Real, int Dims);
template <typename IndexType, typename Real, int Dims>
void THSyclTensor_scatterKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {

  auto queue         = c10::sycl::syclGetCurrentQueue();
  IndexType group_size = (IndexType)c10::sycl::syclMaxWorkGroupSize(queue);
  auto num_groups    = THSYCLCeilDiv(totalElements, group_size);
  auto total_items   = num_groups * group_size;

  auto cgf = DP_Q_CGF(cgh) {
    auto acc_out = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, tensor.data);
    auto acc_src = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, src.data);
    auto acc_index = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, index.data);
    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto tensor_ptr = acc_out.template get_pointer<Real>();
      auto src_ptr = acc_src.template get_pointer<Real>();
      auto index_ptr = acc_index.template get_pointer<int64_t>();
      for (IndexType linearIndex = (IndexType)item.get_global_id(0);
           linearIndex < totalElements; linearIndex += (IndexType)item.get_global_range()[0]) {
        IndexType tensorOffset = 0;
        IndexType srcOffset = 0;
        IndexType indexOffset = 0;

        IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearIndex, dim,
                                                          index, &indexOffset,
                                                          src, &srcOffset,
                                                          tensor, &tensorOffset);
                                                          
        int64_t indexValue = index_ptr[indexOffset];
        // assert(indexValue >= 0 && indexValue < src.sizes[dim]);
        tensorOffset += indexValue * tensor.strides[dim];

        tensor_ptr[tensorOffset] = src_ptr[srcOffset];
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(scatterSycl, IndexType, Real, Dims)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };

  DP_Q_ASYNC_SUBMIT(queue, cgf);  

}

DP_DEF_K2(scatterAddSycl, typename IndexType, typename Real, int Dims);
template <typename IndexType, typename Real, int Dims>
void THSyclTensor_scatterAddKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  
  auto queue         = c10::sycl::syclGetCurrentQueue();
  IndexType group_size = (IndexType)c10::sycl::syclMaxWorkGroupSize(queue);
  auto num_groups    = THSYCLCeilDiv(totalElements, group_size);
  auto total_items   = num_groups * group_size;

  auto cgf = DP_Q_CGF(cgh) {
    auto acc_out = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, tensor.data);
    auto acc_src = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, src.data);
    auto acc_index = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, index.data);
    auto kfn = DP_Q_KFN(DP::nd_item<1>item) {
      auto tensor_ptr = acc_out.template get_pointer<Real>();
      auto src_ptr = acc_src.template get_pointer<Real>();
      auto index_ptr = acc_index.template get_pointer<int64_t>();
    
      for (IndexType linearIndex = (IndexType)item.get_global_id(0);
           linearIndex < totalElements; linearIndex += (IndexType)item.get_global_range()[0]) {
        IndexType tensorOffset = 0;
        IndexType srcOffset = 0;
        IndexType indexOffset = 0;

        IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearIndex, dim,
                                                          index, &indexOffset,
                                                          src, &srcOffset,
                                                          tensor, &tensorOffset);
        
        int64_t indexValue = index_ptr[indexOffset];
        // assert(indexValue >= 0 && indexValue < src.sizes[dim]);
        tensorOffset += indexValue * tensor.strides[dim];

        atomicAdd(&tensor_ptr[tensorOffset], src_ptr[srcOffset]);
      }
    };

    // kick off kernel
    cgh.parallel_for<DP_K(scatterAddSycl, IndexType, Real, Dims)>(
      DP::nd_range<1>(DP::range<1>(total_items), DP::range<1>(group_size)), kfn);
  };
  DP_Q_ASYNC_SUBMIT(queue, cgf);
}


#include <THDP/generic/THSYCLTensorScatterGather.cpp>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorScatterGather.cpp>
#include <THDP/THSYCLGenerateBoolType.h>
