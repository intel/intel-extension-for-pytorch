#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMath.cpp"
#else

void THSYCLTensor_(fill)(THSYCLState *state, THSYCLTensor *self_, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 1, self_));
  at::sycl::SYCL_tensor_apply1<scalar_t>(
    THTensor_wrap(self_), TensorFillOp<scalar_t>(value));

}

void THSYCLTensor_(zero)(THSYCLState *state, THSYCLTensor *self_)
{
  if (THSYCLTensor_(isContiguous)(state, self_)) {
    c10::sycl::syclMemsetAsync(THSYCLTensor_(data)(state, self_),
                    0,
                    sizeof(scalar_t)*THSYCLTensor_(nElement)(state, self_));
  } else {
    THError("THSYCL only support contiguous for zero");
  }
}

void THSYCLTensor_(diag)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src_, int64_t k)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src_));
  int nDimension = THSYCLTensor_(nDimensionLegacyNoScalars)(state, src_);
  THArgCheck((nDimension == 2) || (nDimension == 1), 1, "expected a matrix or a vector");
  if (nDimension == 2) {
    int64_t stride0 = THSYCLTensor_(stride)(state, src_, 0);
    int64_t stride1 = THSYCLTensor_(stride)(state, src_, 1);
    int64_t size0 = THSYCLTensor_(size)(state, src_, 0);
    int64_t size1 = THSYCLTensor_(size)(state, src_, 1);
    int64_t size = (k > 0) ? cl::sycl::min((int64_t)size0, (int64_t)size1 - k) : cl::sycl::min((int64_t)size0 + k, (int64_t)size1);
    THSYCLTensor_(resize1d)(state, self_, size);
    if (size > 0) {
      int64_t strideSelf = THSYCLTensor_(stride)(state, self_, 0);
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      static const auto write_mode = cl::sycl::access::mode::discard_write;
      static const auto read_mode = cl::sycl::access::mode::read;
      auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
      int64_t rng, grng, tile_size;
      c10::sycl::parallel_for_setup(self_->numel(), tile_size, rng, grng);
      sycl_queue.submit([&](cl::sycl::handler& cgh) {
        auto in_acc = c10::sycl::SYCLAccessor<read_mode>(
            cgh, src_->data<scalar_t>());
        auto out_acc = c10::sycl::SYCLAccessor<write_mode>(
            cgh, self_->data<scalar_t>());
        cgh.parallel_for<diag_from_sycl_ker<scalar_t>>(
            cl::sycl::nd_range<1>(cl::sycl::range<1>(grng), cl::sycl::range<1>(tile_size)),
            [=](cl::sycl::nd_item<1> item) {
              size_t id = item.get_global_linear_id();
              auto in_ptr = in_acc.template get_pointer<scalar_t>();
              auto out_ptr = out_acc.template get_pointer<scalar_t>();
              const int64_t bOffset = start + (stride0 + stride1) * id;
              out_ptr[strideSelf * id] = in_ptr[bOffset];
            });
      });
    }
  } else {
    int64_t totalElements = THSYCLTensor_(nElement)(state, src_);
    int64_t size = (k > 0) ? totalElements + k : totalElements - k;
    int64_t strideSrc = THTensor_strideLegacyNoScalars(src_, 0);
    THSYCLTensor_(resize2d)(state, self_, size, size);
    THSYCLTensor_(zero)(state, self_);
    if (size > 0) {
      int64_t stride0 = THSYCLTensor_(stride)(state, self_, 0);
      int64_t stride1 = THSYCLTensor_(stride)(state, self_, 1);
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      static const auto write_mode = cl::sycl::access::mode::discard_write;
      static const auto read_mode = cl::sycl::access::mode::read;
      auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
      int64_t rng, grng, tile_size;
      c10::sycl::parallel_for_setup(self_->numel(), tile_size, rng, grng);
      sycl_queue.submit([&](cl::sycl::handler& cgh) {
        auto in_acc = c10::sycl::SYCLAccessor<read_mode>(
            cgh, src_->data<scalar_t>());
        auto out_acc = c10::sycl::SYCLAccessor<write_mode>(
            cgh, self_->data<scalar_t>());
        cgh.parallel_for<diag_to_sycl_ker<scalar_t>>(
            cl::sycl::nd_range<1>(
                cl::sycl::range<1>(grng), cl::sycl::range<1>(tile_size)),
            [=](cl::sycl::nd_item<1> item) {
              size_t id = item.get_global_linear_id();
              auto in_ptr = in_acc.template get_pointer<scalar_t>();
              auto out_ptr = out_acc.template get_pointer<scalar_t>();
              const int64_t aOffset = start + (stride0 + stride1) * id;
              out_ptr[aOffset] = in_ptr[strideSrc * id];
            });
      });
    }
  }

}

void THSYCLTensor_(nonzero)(THSYCLState* state,
    THSyclLongTensor *tensor, THSYCLTensor *self) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 1, self));
  THSYCLAssertSameGPU(THSyclLongTensor_checkGPU(state, 1, tensor));

  self = THSYCLTensor_(newContiguous)(state, self);

  int64_t num_dim = THSYCLTensor_(nDimensionLegacyNoScalars)(state, self);
  int64_t N = THSYCLTensor_(nElement)(state, self);

  // First to resize out tensor to full elements row
  THSyclLongTensor_resize2d(state, tensor, N, num_dim);
  tensor = THSyclLongTensor_newContiguous(state, tensor);
  // Prepare input tensor strides for calculating result index

  if (N > 0)
  {
    if (THSYCLTensor_canUse32BitIndexMath(state, self)) {
      TensorInfo<scalar_t, uint32_t> input =
              getTensorInfo<scalar_t, THSYCLTensor, uint32_t>(state, self);
      auto idx_fuc = idx_functor<uint32_t>(input);
      input.collapseDims();

      TensorInfo<long, uint32_t> output =
              getTensorInfo<long, THSyclLongTensor, uint32_t>(state, tensor);
      output.collapseDims();

      auto queue = c10::sycl::syclGetCurrentQueue();
      auto num_nonzeros = pattern_scan(queue, input, output, static_cast<uint32_t>(N),
          NonZeroOp<scalar_t>{},
          idx_fuc);

      // Resize the output tensor to the real size
      THSyclLongTensor_resize2d(state, tensor, num_nonzeros, num_dim);
    }
    else {
      TensorInfo<scalar_t, uint64_t> input =
              getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, self);
      auto idx_fuc = idx_functor<uint64_t>(input);
      input.collapseDims();

      TensorInfo<long, uint64_t> output =
              getTensorInfo<long, THSyclLongTensor, uint64_t>(state, tensor);
      output.collapseDims();

      auto queue = c10::sycl::syclGetCurrentQueue();
      auto num_nonzeros = pattern_scan(queue, input, output, static_cast<uint64_t>(N),
          NonZeroOp<scalar_t>{},
          idx_fuc);

      // Resize the output tensor to the real size
      THSyclLongTensor_resize2d(state, tensor, num_nonzeros, num_dim);
    }
  }

  THSYCLTensor_(free)(state, self);
  THSyclLongTensor_free(state, tensor);
}

accreal THSYCLTensor_(trace)(THSYCLState *state, THSYCLTensor *src_) {
  AT_ERROR("not implemented THSYCLTensor_trace\n");
}
#endif
