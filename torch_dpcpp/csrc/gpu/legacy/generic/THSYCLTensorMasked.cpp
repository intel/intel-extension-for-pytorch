#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMasked.cpp"
#else

void THSYCLTensor_(maskedFill)(THSYCLState* state,
    THSYCLTensor* tensor, THSyclByteTensor* mask, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, tensor, mask));
  THArgCheck(THSYCLTensor_(nElement)(state, tensor) ==
             THSyclByteTensor_nElement(state, mask),
             2, "sizes do not match");

  at::sycl::SYCL_tensor_apply2<scalar_t, uint8_t>(THTensor_wrap(tensor),
      THTensor_wrap(mask), TensorMaskedFillOp<scalar_t, unsigned char>(value));
}

void THSYCLTensor_(maskedSelectBool)(THSYCLState* state, THSYCLTensor* tensor,
                                 THSYCLTensor* src, THSyclBoolTensor* mask) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, tensor, src, mask));
  THArgCheck(THSyclBoolTensor_nElement(state, mask) ==
             THSYCLTensor_(nElement)(state, src),
             2, "sizes do not match");

  // Determine our output size
  ptrdiff_t totalElements = THSyclBoolTensor_sumall(state, mask);
  if (totalElements == 0) {
    THSYCLTensor_(resize1d)(state, tensor, totalElements);
    return ;
  }

  THSYCLTensor* tensorContig = THSYCLTensor_(newContiguous)(state, tensor);

  THSYCLTensor_(resize1d)(state, tensorContig, totalElements);
  if (tensor != tensorContig) {
    THSYCLTensor_(resize1d)(state, tensor, totalElements);
  }

  THSyclLongTensor* maskLong = THSyclLongTensor_new(state);
  at::IntArrayRef maskSizes = mask->sizes();
  THSyclLongTensor_resize(state, maskLong, maskSizes, {});
  THSYCLTensor_(copy)(state, maskLong, mask);


  // Use a prefix sum to determine the output locations of the masked elements
  THSyclLongTensor* maskPrefixSum = THSyclLongTensor_new(state);
  THSyclLongTensor_resize(state, maskPrefixSum, maskSizes, {});

  auto maskLong_data = maskLong->data();
  auto maskLong_size = maskLong->numel() * (maskLong->dtype().itemsize());
  auto maskPrefixSum_data = maskPrefixSum->data();
  auto maskPrefixSum_size = maskPrefixSum->numel() * (maskPrefixSum->dtype().itemsize());
  int64_t size = THSyclLongTensor_nElement(state, maskLong);

  auto sycl_queue = c10::sycl::syclGetCurrentQueue();
  int64_t rng, GRange, tileSize;
  c10::sycl::parallel_for_setup(size, tileSize, rng, GRange);

  // command group functions
  auto cgf = DP_Q_CGF(cgh) {
    auto acc_maskLong = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, maskLong_data, maskLong_size);
    auto acc_maskPrefixSum = c10::sycl::SYCLAccessor<dp_discard_w_mode>(cgh, maskPrefixSum_data, maskPrefixSum_size);

    // kernel function per work-item
    auto kfn = DP_Q_KFN() {
      dp_global_ptr_cpt<int64_t> maskLong_ptr = acc_maskLong.template get_pointer<int64_t>();
      dp_global_ptr_pt<int64_t> maskPrefixSum_ptr = acc_maskPrefixSum.template get_pointer<int64_t>();
      sycl_inclusive_scan(maskLong_ptr, maskLong_ptr + size, maskPrefixSum_ptr, AddOp<int64_t>());
    };
    // kick off kernel
    // (TODO) single_task need replaced due to low efficiency
    cgh.single_task<DP_K(maskedSelect_scan_sycl_ker, scalar_t)>(kfn);
  };

    // submit to SYCL queue
    DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);


    TensorInfo<scalar_t, uint64_t> src_info =
            getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, src);
    src_info.collapseDims();

    TensorInfo<bool, uint64_t> mask_info =
            getTensorInfo<bool, THSyclBoolTensor, uint64_t>(state, mask);
    mask_info.collapseDims();

  // command group function
  auto cgfMaskedSelect = DP_Q_CGF(cgh) {
    auto acc_src = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, src->data<scalar_t>());
    auto acc_mask = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, mask->data<bool>());
    auto acc_maskPrefixSum = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, maskPrefixSum->data<int64_t>());
    auto acc_tensor = c10::sycl::SYCLAccessor<dp_discard_w_mode>(cgh, tensorContig->data<scalar_t>());

    // kernel function per work-item
    auto kfn = DP_Q_KFN(DP::nd_item<1> item){
      int64_t linear_index = item.get_global_linear_id();

      dp_global_ptr_pt<scalar_t> src_ptr = acc_src.template get_pointer<scalar_t>();
      dp_global_ptr_pt<bool> mask_ptr = acc_mask.template get_pointer<bool>();
      dp_global_ptr_pt<int64_t> maskPrefix_ptr = acc_maskPrefixSum.template get_pointer<int64_t>();
      dp_global_ptr_pt<scalar_t> tensor_ptr = acc_tensor.template get_pointer<scalar_t>();

      if (linear_index < size) {
          // The mask tensor maybe not contiguos.
          auto mask_offset = IndexToOffset<bool, uint64_t>().get(linear_index, mask_info);
        if (mask_ptr[mask_offset]) {
          // The src tensor maybe not contiguos.
          auto src_offset = IndexToOffset<scalar_t, uint64_t>().get(linear_index, src_info);
          tensor_ptr[maskPrefix_ptr[linear_index] - 1] = src_ptr[src_offset];
        }
      }
    };
    cgh.parallel_for<DP_K(TensorMaskedSelectOp, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(GRange), DP::range<1>(tileSize)), kfn);
  };

  // submit to SYCL queue
  DP_Q_ASYNC_SUBMIT(sycl_queue, cgfMaskedSelect);

  THSyclLongTensor_free(state, maskLong);
  THSyclLongTensor_free(state, maskPrefixSum);

  if (tensor != tensorContig) {
    THSYCLTensor_(freeCopyTo)(state, tensorContig, tensor);
  } else {
    THSYCLTensor_(free)(state, tensorContig);
  }

}

void THSYCLTensor_(maskedFillBool)(THSYCLState* state,
    THSYCLTensor* tensor, THSyclBoolTensor* mask, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, tensor, mask));
  THArgCheck(THSYCLTensor_(nElement)(state, tensor) ==
             THSyclByteTensor_nElement(state, mask),
             2, "sizes do not match");

  at::sycl::SYCL_tensor_apply2<scalar_t, bool>(THTensor_wrap(tensor),
      THTensor_wrap(mask), TensorMaskedFillOp<scalar_t, bool>(value));

}

void THSYCLTensor_(maskedCopy)(THSYCLState* state,
                               THSYCLTensor *tensor, THSyclByteTensor *mask, THSYCLTensor *src)
{

  AT_ERROR("not implemented THSYCLTensor_maskedCopy\n");

}

void THSYCLTensor_(maskedCopyBool)(THSYCLState* state,
    THSYCLTensor* tensor, THSyclBoolTensor* mask, THSYCLTensor* src) {

  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, tensor, src, mask));
  auto maskSize = THSyclByteTensor_nElement(state, mask);
  auto tensorSize = THSYCLTensor_(nElement)(state, tensor);
  auto srcSize = THSYCLTensor_(nElement)(state, src);

  // `mask` and `tensor` must have the same number of elements
  THArgCheck(maskSize == tensorSize, 2,
             "mask and tensor must have the same number of elements");

  // Determine our output size
  auto totalElements = THSyclBoolTensor_sumall(state, mask);

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (totalElements > srcSize) {
    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
  }

  THSyclLongTensor* maskLong = THSyclLongTensor_new(state);
  at::IntArrayRef maskSizes = mask->sizes();
  THSyclLongTensor_resize(state, maskLong, maskSizes, {});
  THSYCLTensor_(copy)(state, maskLong, mask);

  // Use a prefix sum to determine the output locations of the masked elements
  THSyclLongTensor* maskPrefixSum = THSyclLongTensor_new(state);
  THSyclLongTensor_resize(state, maskPrefixSum, maskSizes, {});

  auto maskLong_data = maskLong->data();
  auto maskLong_size = maskLong->numel() * (maskLong->dtype().itemsize());
  auto maskPrefixSum_data = maskPrefixSum->data();
  auto maskPrefixSum_size = maskPrefixSum->numel() * (maskPrefixSum->dtype().itemsize());
  auto size = THSyclLongTensor_nElement(state, maskLong);

  auto sycl_queue = c10::sycl::syclGetCurrentQueue();
  int64_t rng, GRange, tileSize;
  c10::sycl::parallel_for_setup(size, tileSize, rng, GRange);

  // command group functions
  auto cgf = DP_Q_CGF(cgh) {
    auto acc_maskLong = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, maskLong_data, maskLong_size);
    auto acc_maskPrefixSum = c10::sycl::SYCLAccessor<dp_discard_w_mode>(cgh, maskPrefixSum_data, maskPrefixSum_size);

    // kernel function
    auto kfn = DP_Q_KFN() {
      dp_global_ptr_cpt<int64_t> maskLong_ptr = acc_maskLong.template get_pointer<int64_t>();
      dp_global_ptr_pt<int64_t> maskPrefixSum_ptr = acc_maskPrefixSum.template get_pointer<int64_t>();
      sycl_exclusive_scan(maskLong_ptr, maskLong_ptr + size, maskPrefixSum_ptr, static_cast<int64_t>(0), AddOp<int64_t>());
    };
    // (TODO) single_task need replaced due to low efficiency
    cgh.single_task<DP_K(maskedCopy_scan_sycl_ker, scalar_t)>(kfn);
  };

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  THSYCLTensor* contigSrc = THSYCLTensor_(newContiguous)(state, src);

  // submit to SYCL queue
  DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);

  // command group function
  // copy src to tensor according to mask
  auto cgfMaskedCopy = DP_Q_CGF(cgh) {
    auto acc_src = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, contigSrc->data<scalar_t>());
    auto acc_mask = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, mask->data<bool>());
    auto acc_maskPrefixSum = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, maskPrefixSum->data<int64_t>());
    auto acc_tensor = c10::sycl::SYCLAccessor<dp_discard_w_mode>(cgh, tensor->data<scalar_t>());

    // kernel function
    auto kfn = DP_Q_KFN(DP::nd_item<1> item){
      int64_t linear_index = item.get_global_linear_id();
      dp_global_ptr_pt<scalar_t> src_ptr = acc_src.template get_pointer<scalar_t>();
      dp_global_ptr_pt<bool> mask_ptr = acc_mask.template get_pointer<bool>();
      dp_global_ptr_pt<int64_t> maskPrefix_ptr = acc_maskPrefixSum.template get_pointer<int64_t>();
      dp_global_ptr_pt<scalar_t> tensor_ptr = acc_tensor.template get_pointer<scalar_t>();
      if (linear_index < size) {
        if (mask_ptr[linear_index]) {
          tensor_ptr[linear_index] = src_ptr[maskPrefix_ptr[linear_index]];
        }
      }
    };

    cgh.parallel_for<DP_K(TensorMaskedCopyOp, scalar_t)>(
      DP::nd_range<1>(DP::range<1>(GRange), DP::range<1>(tileSize)), kfn);
  };

  // submit to SYCL queue
  DP_Q_ASYNC_SUBMIT(sycl_queue, cgfMaskedCopy);

  THSYCLTensor_(free)(state, contigSrc);

  THSyclLongTensor_free(state, maskLong);

  THSyclLongTensor_free(state, maskPrefixSum);

}

void THSYCLTensor_(maskedSelect)(THSYCLState* state,
    THSYCLTensor* tensor, THSYCLTensor* src, THSyclByteTensor* mask) {
  AT_ERROR("not implemented THSYCLTensor_maskedSelect\n");
}
#endif
