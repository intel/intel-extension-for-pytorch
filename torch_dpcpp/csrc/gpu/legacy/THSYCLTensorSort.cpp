#include <legacy/THSYCLTensor.hpp>
#include <legacy/THSYCLSortUtils.h>
#include <legacy/THSYCLTensorMath.h>
#include <legacy/THSYCLTensorCopy.h>
#include <legacy/THSYCLTensorSort.h>
#include <legacy/generic/THSYCLTensorSort.cpp>
#include <legacy/THSYCLGenerateAllTypes.h>
#include <legacy/generic/THSYCLTensorSort.cpp>
#include <legacy/THSYCLGenerateBoolType.h>

template <typename IndexType, int Dim>
void fillSliceWithIndex(TensorInfo<int64_t, IndexType> out,
                   IndexType totalSlices,
                   IndexType sliceSize,
                   IndexType sliceStride) {
  auto &sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  int64_t local_size = sycl_queue.get_device(). template get_info<cl::sycl::info::device::max_work_group_size>();
  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    auto out_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, out.data);

    cgh.parallel_for<fill_slice_sycl_ker<IndexType, Dim>>(
        cl::sycl::nd_range<1>(cl::sycl::range<1>(totalSlices*local_size), cl::sycl::range<1>(local_size)),
        [=](cl::sycl::nd_item<1> item_id) {
      IndexType local_id = item_id.get_local_id(0);
      IndexType slice = item_id.get_group_linear_id();
      const uint64_t offset =
        IndexToOffset<int64_t, IndexType, Dim>::get(slice, out);
      int64_t* base = out_acc.template get_pointer<int64_t>() + offset;

      for (IndexType i = local_id; i < sliceSize; i += item_id.get_local_range(0)) {
        // Torch indices are 1-based (hence the +1)
        base[i * sliceStride] = i /* + TH_INDEX_BASE */;
      }
    });
  });
}

void THSyclLongTensor_fillSliceWithIndex(THSYCLState* state,
                                         THSyclLongTensor* t,
                                         int dim) {
  int64_t dims = THSyclLongTensor_nDimensionLegacyNoScalars(state, t);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 2, SYCLTORCH_DIM_WARNING);

  ptrdiff_t inElements = THSyclLongTensor_nElement(state, t);
  if (inElements > 0) {
    int64_t sliceSize = THSyclLongTensor_sizeLegacyNoScalars(state, t, dim);
    ptrdiff_t numSlices = inElements / sliceSize;

#define FILL_INDEX(T, DIM)                                         \
    fillSliceWithIndex<T, DIM>(                                     \
        info, numSlices, sliceSize, info.strides[collapseDim])

    if (THSYCLTensor_canUse32BitIndexMath(state, t)) {
      TensorInfo<int64_t, uint32_t> info =
        getTensorInfo<int64_t, THSyclLongTensor, unsigned int>(state, t);
      info.reduceDim(dim);
      int collapseDim = info.collapseDims(dim);
      if (info.isContiguous()) {
        FILL_INDEX(unsigned int, -2);
      } else {
        if (info.dims == 1) {
          FILL_INDEX(unsigned int, 1);
        } else if (info.dims == 2) {
          FILL_INDEX(unsigned int, 2);
        } else {
          FILL_INDEX(unsigned int, -1);
        }
      }
    } else {
      TensorInfo<int64_t, uint64_t> info =
        getTensorInfo<int64_t, THSyclLongTensor, uint64_t>(state, t);
      info.reduceDim(dim);
      int collapseDim = info.collapseDims(dim);

      // catch-all implementation
      FILL_INDEX(uint64_t, -1);
    }
  }
}

