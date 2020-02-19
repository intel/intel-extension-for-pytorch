#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorIndex.cpp"
#else

#include <legacy/THSYCLTensorTypeUtils.h>

// TensorInfo
using namespace at::sycl::detail;

void THSYCLTensor_(indexSelect)(THSYCLState *state,
    THSYCLTensor *dst, THSYCLTensor *src, int dim, THSyclLongTensor *indices)
{
//    TODO: CUDA check to sycl
//    THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, dst, src, indices));
    int dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, dst);
    THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 2, SYCLTORCH_DIM_WARNING);
    dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, src);
    THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 3, SYCLTORCH_DIM_WARNING);
    dims = THSyclLongTensor_nDimensionLegacyNoScalars(state, indices);
    THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 5, SYCLTORCH_DIM_WARNING);

    int srcDims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, src);
    THArgCheck(THSyclLongTensor_nDimensionLegacyNoScalars(state, indices) <= 1, 3,
             "Index is supposed to be an empty tensor or a vector");
    THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
    THArgCheck(srcDims > 0, 2, "Source tensor is empty");

    TensorInfo<int64_t, unsigned int> indices_info =
      getTensorInfo<int64_t, THSyclLongTensor, unsigned int>(state, indices);
    indices_info.collapseDims();

//      TODO:change to THSYCLTensor_(xxx)
//    auto newSize = THSYCLTensor_(sizesLegacyNoScalars)(state, src);
    auto new_size = THSYCLTensor_sizesLegacyNoScalars(state, src);
    new_size[dim] = indices_info.sizes[0];
    THSYCLTensor_(resize)(state, dst, new_size, {});

    ptrdiff_t dst_num_elem = THSYCLTensor_(nElement)(state, dst);
    if (dst_num_elem == 0) {
        return;
    }

    TensorInfo<scalar_t, unsigned int> dst_info =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, dst);
    int dst_select_dim = dst_info.collapseDims(dim);
    dst_info.reduceDim(dst_select_dim);

    TensorInfo<scalar_t, unsigned int> src_info =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, src);
    int src_select_dim = src_info.collapseDims(dim);
    src_info.reduceDim(src_select_dim);

    // The `src` is partitioned into two parts:
    // -the size of each slice we are indexing, which is the
    // total size of the tensor ignoring dimension `dim`;
    // -the number of indices we are choosing, which is the total size
    // of the tensor `indices`.
    // TODO: if the slice number is to large. Need to balance the work group and work item number.
    // Make the work balance based on the MCU number.
    // auto __mcu = sycl_queue.get_device().template get_info<dp_dev_max_units>();
    uint64_t num_slices = THSyclLongTensor_nElement(state, indices);;

    auto slice_size = dst_num_elem / num_slices;

    auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();

    auto wgroup_size = sycl_queue.get_device().template \
            get_info<dp_dev_max_wgroup_size>();

    wgroup_size = std::min(decltype(wgroup_size)(slice_size), wgroup_size);

    auto n_work_item_iter = (slice_size + wgroup_size - 1) / wgroup_size;

    auto src_data = src->data();
    auto dst_data = dst->data();
    auto idx_data = indices->data();
    auto src_size = src->storage().numel() * \
            (src->dtype().itemsize());
    auto dst_size = dst->storage().numel() * \
            (dst->dtype().itemsize());
    auto idx_size = indices->storage().numel() * \
            (indices->dtype().itemsize());

    auto cgf = DP_Q_CGF(__cgh) {
        auto src_acc = c10::sycl::SYCLAccessor<dp_r_mode>(__cgh, src_data, src_size);
        auto dst_acc = c10::sycl::SYCLAccessor<dp_discard_w_mode>(__cgh, dst_data, dst_size);
        auto idx_acc = c10::sycl::SYCLAccessor<dp_r_mode>(__cgh, idx_data, idx_size);

        __cgh.parallel_for_work_group<DP_K(index_select_ker, scalar_t)>(
            DP::range</*dim=*/1>(num_slices),
            DP::range</*dim=*/1>(wgroup_size),
            [=](DP::group<1> group_id) {
                auto src_ptr = src_acc.template get_pointer<scalar_t>();
                auto dst_ptr = dst_acc.template get_pointer<scalar_t>();
                auto idx_ptr = idx_acc.template get_pointer<long>();

                auto dst_slice_id = group_id.get_id()[0];

                auto slice_off = IndexToOffset<int64_t, unsigned int>::get(dst_slice_id, indices_info);
                auto src_slice_id = idx_ptr[slice_off]/* - TH_INDEX_BASE*/;

                auto g_src_ptr = src_ptr + src_slice_id * src_info.strides[src_select_dim];
                auto g_dst_ptr = dst_ptr + dst_slice_id * dst_info.strides[dst_select_dim];

                group_id.parallel_for_work_item([=](DP::h_item<1> item_id) {

                    auto ii_ = item_id.get_logical_local_id()[0];
                    auto src_offset_ =
                            IndexToOffset<scalar_t, unsigned int>::get(ii_, src_info);
                    auto dst_offset_ =
                            IndexToOffset<scalar_t, unsigned int>::get(ii_, dst_info);

                    g_dst_ptr[ dst_offset_ ] = g_src_ptr[ src_offset_ ];

                    for (decltype(n_work_item_iter) iter = 1; iter < n_work_item_iter;iter++)
                    {
                        auto __inner_idx = iter * wgroup_size + ii_;
                        if (__inner_idx < slice_size)
                        {
                            src_offset_ = IndexToOffset<scalar_t, unsigned int>::get(__inner_idx, src_info);
                            dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(__inner_idx, dst_info);

                            g_dst_ptr[ dst_offset_ ] = g_src_ptr[ src_offset_ ];
                        }
                    }
                });
            }
        );
    };

    DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
  return;
}

void THSYCLTensor_(indexCopy)(THSYCLState* state, THSYCLTensor* dst,
    int dim, THSyclLongTensor* indices, THSYCLTensor* src) {
  AT_ERROR("not implemented THSYCLTensor_indexCopy\n");
}

void THSYCLTensor_(take)(THSYCLState* state, THSYCLTensor* dst,
    THSYCLTensor* src, THSyclLongTensor* index) {
  AT_ERROR("not implemented THSYCLTensor_take\n");
}

void THSYCLTensor_(indexAdd)(THSYCLState* state, THSYCLTensor* dst,
    int dim, THSyclLongTensor* indices, THSYCLTensor* src) {
  int dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, dst);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 2, SYCLTORCH_DIM_WARNING);
  dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, src);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 5, SYCLTORCH_DIM_WARNING);
  dims = THSyclLongTensor_nDimensionLegacyNoScalars(state, indices);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 4, SYCLTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int dstDims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, dst);
  int srcDims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, src);

  THArgCheck(THSyclLongTensor_nDimensionLegacyNoScalars(state, indices) == 1, 4,
             "expecting vector of indices");
  THArgCheck(dim >= 0 && dim < dstDims, 2, "Indexing dim is out of bounds");

  ptrdiff_t dstSliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      dstSliceSize *= THTensor_sizeLegacyNoScalars(dst, d);
    }
  }
  
  THArgCheck(dim < srcDims, 3, "Indexing dim is out of bounds");
  THArgCheck(THSyclLongTensor_nElement(state, indices) == THTensor_sizeLegacyNoScalars(src, dim), 4,
             "length of src.size[dim] is not equal to length of indices");
  
  ptrdiff_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims) mismatch = true;

  for (int d = 0; d < srcDims; d++) {
    if (d != dim) {
      srcSliceSize *= THTensor_sizeLegacyNoScalars(src, d);
      if (!mismatch && THTensor_sizeLegacyNoScalars(dst, d) != THTensor_sizeLegacyNoScalars(src, d)) mismatch = true;
    }
  }

  THArgCheck(dstSliceSize == srcSliceSize, 2,
             "Source/destination tensor have different slice sizes (%ld vs %ld)",
             dstSliceSize, srcSliceSize);
  
  if (mismatch) {
    static bool warningShown = false;
    if (!warningShown) {
      warningShown = true;
      fprintf(stderr,
              "Warning: source/destination slices have same size but different "
              "shape for an index operation. This behavior is deprecated.\n");
    }
  }
  ptrdiff_t sliceSize = dstSliceSize;

  ptrdiff_t srcTotalSize = THSYCLTensor_(nElement)(state, src);
  int64_t dstAddDimSize = THSYCLTensor_(sizeLegacyNoScalars)(state, dst, dim);
  ptrdiff_t numIndices = THSyclLongTensor_nElement(state, indices);

  if (sliceSize == 0) {
    return;
  }

  TensorInfo<int64_t, unsigned int> indices_info =
      getTensorInfo<int64_t, THSyclLongTensor, unsigned int>(state, indices);
  indices_info.collapseDims();
  
  TensorInfo<scalar_t, unsigned int> dst_info =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, dst);
  int dst_add_dim = dst_info.collapseDims(dim);
  dst_info.reduceDim(dst_add_dim);

  TensorInfo<scalar_t, unsigned int> src_info =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, src);
  int src_add_dim = src_info.collapseDims(dim);
  src_info.reduceDim(src_add_dim);

  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();

  auto wgroup_size = sycl_queue.get_device().template \
          get_info<dp_dev_max_wgroup_size>();

  wgroup_size = std::min(decltype(wgroup_size)(sliceSize), wgroup_size);

  auto n_work_item_iter = (sliceSize + wgroup_size - 1) / wgroup_size;

  auto src_data = src->data();
  auto dst_data = dst->data();
  auto idx_data = indices->data();
  auto src_size = src->storage().numel() * \
          (src->dtype().itemsize());
  auto dst_size = dst->storage().numel() * \
          (dst->dtype().itemsize());
  auto idx_size = indices->storage().numel() * \
          (indices->dtype().itemsize());

  auto cgf = DP_Q_CGF(__cgh) {
    auto src_acc = c10::sycl::SYCLAccessor<dp_r_mode>(__cgh, src_data, src_size);
    auto dst_acc = c10::sycl::SYCLAccessor<dp_discard_w_mode>(__cgh, dst_data, dst_size);
    auto idx_acc = c10::sycl::SYCLAccessor<dp_r_mode>(__cgh, idx_data, idx_size);

    __cgh.parallel_for_work_group<DP_K(index_add_ker, scalar_t)>(
      DP::range</*dim=*/1>(numIndices),
      DP::range</*dim=*/1>(wgroup_size),
      [=](DP::group<1> group_id) {
        auto src_ptr = src_acc.template get_pointer<scalar_t>();
        auto dst_ptr = dst_acc.template get_pointer<scalar_t>();
        auto idx_ptr = idx_acc.template get_pointer<long>();

        auto dst_slice_id = group_id.get_id()[0];
        //auto slice_off = IndexToOffset<int64_t, unsigned int>::get(dst_slice_id, indices_info);
        auto g_idx_ptr = idx_ptr;
        auto g_dst_ptr = dst_ptr + g_idx_ptr[ dst_slice_id ] * dst_info.strides[dst_add_dim];
        auto g_src_ptr = src_ptr + dst_slice_id * src_info.strides[src_add_dim];

        group_id.parallel_for_work_item([=](DP::h_item<1> item_id) {

          auto ii_ = item_id.get_logical_local_id()[0];
          auto dst_offset_ =
                  IndexToOffset<scalar_t, unsigned int>::get(ii_, dst_info);
          auto src_offset_ = 
                  IndexToOffset<scalar_t, unsigned int>::get(ii_, src_info);
          g_dst_ptr[ dst_offset_ ] += g_src_ptr[ src_offset_ ];

          for (decltype(n_work_item_iter) iter = 1; iter < n_work_item_iter;iter++)
          {
            auto idx_offset_ = 
                  IndexToOffset<int64_t, unsigned int>::get(iter, indices_info);
            auto __inner_idx = g_idx_ptr[ idx_offset_ ] * wgroup_size + ii_;
            auto __src_idx = idx_offset_ * wgroup_size + ii_;

            if (__src_idx < srcTotalSize)
            {
              dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(__inner_idx, dst_info);
              src_offset_ = IndexToOffset<scalar_t, unsigned int>::get(__src_idx, src_info);
              g_dst_ptr[ dst_offset_ ] += g_src_ptr[ src_offset_ ];
            }
          }
        });
      }
    );
  };

  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

void THSYCLTensor_(indexFill)(THSYCLState* state, THSYCLTensor* dst,
    int dim, THSyclLongTensor* indices, scalar_t val) {
  int dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, dst);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 2, SYCLTORCH_DIM_WARNING);
  dims = THSyclLongTensor_nDimensionLegacyNoScalars(state, indices);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 4, SYCLTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int dstDims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, dst);
  //int srcDims = dstDims;

  THArgCheck(THSyclLongTensor_nDimensionLegacyNoScalars(state, indices) == 1, 4,
             "expecting vector of indices");
  THArgCheck(dim >= 0 && dim < dstDims, 2, "Indexing dim is out of bounds");

  ptrdiff_t sliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      sliceSize *= THTensor_sizeLegacyNoScalars(dst, d);
    }
  }
  ptrdiff_t dstTotalSize = THSYCLTensor_(nElement)(state, dst);
  int64_t dstFillDimSize = THSYCLTensor_(sizeLegacyNoScalars)(state, dst, dim);
  ptrdiff_t numIndices = THSyclLongTensor_nElement(state, indices);

  if (sliceSize == 0) {
    return;
  }

  TensorInfo<int64_t, unsigned int> indices_info =
      getTensorInfo<int64_t, THSyclLongTensor, unsigned int>(state, indices);
  indices_info.collapseDims();
  
  TensorInfo<scalar_t, unsigned int> dst_info =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, dst);
  int dst_fill_dim = dst_info.collapseDims(dim);
  dst_info.reduceDim(dst_fill_dim);

  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();

  auto wgroup_size = sycl_queue.get_device().template \
          get_info<dp_dev_max_wgroup_size>();

  wgroup_size = std::min(decltype(wgroup_size)(sliceSize), wgroup_size);

  auto n_work_item_iter = (sliceSize + wgroup_size - 1) / wgroup_size;

  auto dst_data = dst->data();
  auto idx_data = indices->data();
  auto dst_size = dst->storage().numel() * \
          (dst->dtype().itemsize());
  auto idx_size = indices->storage().numel() * \
          (indices->dtype().itemsize());

  auto cgf = DP_Q_CGF(__cgh) {
    auto dst_acc = c10::sycl::SYCLAccessor<dp_discard_w_mode>(__cgh, dst_data, dst_size);
    auto idx_acc = c10::sycl::SYCLAccessor<dp_r_mode>(__cgh, idx_data, idx_size);

    __cgh.parallel_for_work_group<DP_K(index_fill_ker, scalar_t)>(
      DP::range</*dim=*/1>(numIndices),
      DP::range</*dim=*/1>(wgroup_size),
      [=](DP::group<1> group_id) {
        auto dst_ptr = dst_acc.template get_pointer<scalar_t>();
        auto idx_ptr = idx_acc.template get_pointer<long>();

        auto dst_slice_id = group_id.get_id()[0];
        //auto slice_off = IndexToOffset<int64_t, unsigned int>::get(dst_slice_id, indices_info);
        auto g_idx_ptr = idx_ptr;
        auto g_dst_ptr = dst_ptr + g_idx_ptr[ dst_slice_id ] * dst_info.strides[dst_fill_dim];

        group_id.parallel_for_work_item([=](DP::h_item<1> item_id) {

          auto ii_ = item_id.get_logical_local_id()[0];
          auto dst_offset_ =
                  IndexToOffset<scalar_t, unsigned int>::get(ii_, dst_info);
          g_dst_ptr[ dst_offset_ ] = val;

          for (decltype(n_work_item_iter) iter = 1; iter < n_work_item_iter;iter++)
          {
            auto idx_offset_ = 
                  IndexToOffset<int64_t, unsigned int>::get(iter, indices_info);
            auto __inner_idx = g_idx_ptr[ idx_offset_ ] * wgroup_size + ii_;

            if (__inner_idx < dstTotalSize)
            {
              dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(__inner_idx, dst_info);

              g_dst_ptr[ dst_offset_ ] = val;
            }
          }
        });
      }
    );
  };

  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}
#endif
