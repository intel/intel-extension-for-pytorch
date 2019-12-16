#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathScan.cpp"
#else

#ifndef THC_REAL_IS_HALF
template<class BinaryFunction>
void THSYCLTensor_(scanThrust)(THSYCLState *state, THSYCLTensor *dst,
                               THSYCLTensor *src, BinaryFunction binary_op) {
  auto dst_data = dst->data();
  auto dst_size = dst->numel() * (dst->dtype().itemsize());
  auto src_data = src->data();
  auto src_size = src->numel() * (src->dtype().itemsize());
  ptrdiff_t size = THSYCLTensor_(nElement)(state, src);

   auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
   sycl_queue.submit([&](cl::sycl::handler &cgh) {
     auto acc_dst = c10::sycl::SYCLAccessor<cl::sycl::access::mode::discard_write>(cgh, dst_data, dst_size);
     auto acc_src = c10::sycl::SYCLAccessor<cl::sycl::access::mode::read>(cgh, src_data, src_size);
     // (TODO) single_task need replaced due to low efficiency
     cgh.single_task<scanthrust_sycl_ker<scalar_t, BinaryFunction>>([=]() {
       auto ptr_dst = acc_dst.template get_pointer<scalar_t>();
       auto ptr_src = acc_src.template get_pointer<scalar_t>();
       sycl_inclusive_scan(ptr_src, ptr_src + size, ptr_dst, binary_op);
     });
   });
}
#endif


template<class BinaryOp>
void THSYCLTensor_(scanOuterDim)(THSYCLState *state, THSYCLTensor *tgt,
                                 THSYCLTensor *src, int dimension,
                                 scalar_t init, BinaryOp binary_op) {
  auto totalElements = tgt->numel();
  auto tgt_data = tgt->data();
  auto tgt_size = totalElements * (tgt->dtype().itemsize());
  auto src_data = src->data();
  auto src_size = totalElements* (src->dtype().itemsize());
  int64_t n = src->size(dimension);
  int64_t stride = src->stride(dimension);
  int64_t batch = totalElements / (n * stride);

  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  int64_t rng, GRange, tileSize;
  c10::sycl::parallel_for_setup(totalElements, tileSize, rng, GRange);
  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    auto src_acc = c10::sycl::SYCLAccessor<cl::sycl::access::mode::read>(cgh, src_data, src_size);
    auto tgt_acc = c10::sycl::SYCLAccessor<cl::sycl::access::mode::discard_write>(cgh, tgt_data, tgt_size);
    cgh.parallel_for<scanOuterDim_sycl_kernel<scalar_t, BinaryOp>>(
        cl::sycl::nd_range<1>(
            cl::sycl::range<1>(tileSize), cl::sycl::range<1>(tileSize)),
        [=](cl::sycl::nd_item<1> item) {
          auto src_ptr = src_acc.template get_pointer<scalar_t>();
          auto tgt_ptr = tgt_acc.template get_pointer<scalar_t>();
          for (int64_t linearIndex = item.get_global_id(0);
               linearIndex < totalElements; linearIndex += item.get_global_range()[0]) {
            int64_t base_start = linearIndex % (batch * stride);
            int64_t start = (base_start / stride) * n * stride + base_start % stride;
            scalar_t result = init;
            for (int j = 0; j < n; ++j) {
              result = binary_op(result, src_ptr[start + j * stride]);
              tgt_ptr[start + j * stride] = result;
            }
          }
        });
  });
}

template<class BinaryFunction>
void THSYCLTensor_(scanInnermostDim)(THSYCLState *state, THSYCLTensor *tgt,
                                     THSYCLTensor *src, scalar_t init,
                                     BinaryFunction binary_op) {
  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();

  auto totalElements = tgt->numel();
  auto tgt_data = tgt->data();
  auto tgt_size = totalElements * (tgt->dtype().itemsize());
  auto src_data = src->data();
  auto src_size = totalElements * (src->dtype().itemsize());
  auto dimension = tgt->dim() - 1;
  int64_t n = src->size(dimension);
  int64_t stride = src->stride(dimension);
  int64_t batch = totalElements / (n * stride);

  int64_t rng, GRange, tileSize;
  c10::sycl::parallel_for_setup(totalElements, tileSize, rng, GRange);

  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    auto src_acc = c10::sycl::SYCLAccessor<cl::sycl::access::mode::read>(cgh, src_data, src_size);
    auto tgt_acc = c10::sycl::SYCLAccessor<cl::sycl::access::mode::discard_write>(cgh, tgt_data, tgt_size);
    cgh.parallel_for<scanInnerDim_sycl_kernel<scalar_t, BinaryFunction>>(
        cl::sycl::nd_range<1>(
            cl::sycl::range<1>(tileSize), cl::sycl::range<1>(tileSize)),
        [=](cl::sycl::nd_item<1> item) {
          auto src_ptr = src_acc.template get_pointer<scalar_t>();
          auto tgt_ptr = tgt_acc.template get_pointer<scalar_t>();

          for (int64_t linearIndex = item.get_global_id(0);
               linearIndex < totalElements; linearIndex += item.get_global_range()[0]) {
            int64_t start = linearIndex % batch * n;
            scalar_t result = init;
            for (int64_t j = 0; j < n; ++j) {
              result = binary_op(result, src_ptr[start + j]);
              tgt_ptr[start + j] = result;
            }
          }
        });
  });
}

template<class BinaryFunction>
void THSYCLTensor_(scanDim)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, int dimension,
                            scalar_t init, BinaryFunction binary_op) {
  int ndim = THSYCLTensor_(nDimensionLegacyNoScalars)(state, src);
  THArgCheck(dimension >= 0 && dimension < ndim, 3, "dimension %d out of range",
      dimension);

  THSYCLTensor_(resizeAs)(state, self_, src);
  THSYCLTensor *self = THSYCLTensor_(newContiguous)(state, self_);
  src = THSYCLTensor_(newContiguous)(state, src);

  if (!self->is_empty()) {
  #ifndef THC_REAL_IS_HALF
    if (ndim == 1) {
      // thrust does not take an "init"
      THSYCLTensor_(scanThrust)(state, self, src, binary_op);
    } else
  #endif
    if (dimension == ndim - 1) {
      THSYCLTensor_(scanInnermostDim)(state, self, src, init, binary_op);
    } else {
      THSYCLTensor_(scanOuterDim)(state, self, src, dimension, init, binary_op);
    }
  }
  THSYCLTensor_(free)(state, src);
  THSYCLTensor_(freeCopyTo)(state, self, self_);
}

void THSYCLTensor_(cumprod)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self, src));
  THSYCLTensor_(scanDim)(state, self, src, dimension,
                           ScalarConvert<float, scalar_t>::to(1.0), MulOp<scalar_t>());
}

void THSYCLTensor_(cumsum)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self, src));
  THSYCLTensor_(scanDim)(state, self, src, dimension,
                          ScalarConvert<float, scalar_t>::to(0.0), AddOp<scalar_t>());
}


#endif
