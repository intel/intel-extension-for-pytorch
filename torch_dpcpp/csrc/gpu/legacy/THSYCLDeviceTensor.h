#ifndef THSYCL_DEVICE_TENSOR_INC
#define THSYCL_DEVICE_TENSOR_INC

template <int Dim, typename data_type, cl::sycl::access::mode AccMode, typename IndexT = int>
class THSYCLDeviceTensor {
 public:
   THSYCLDeviceTensor(THSYCLState* state, THSYCLTensor *t, cl::sycl::handler &cgh) 
    : dim_(Dim), syclAcc_(c10::sycl::SYCLAccessor<AccMode>(cgh, t->data<data_type>())) {
     if (Dim != THSYCLTensor_nDimensionLegacyAll(state, t)) {
       THError("THSyclTensor dimension mismatch");
     }

     for (int i = 0; i < Dim; ++i) {
       int64_t size = THTensor_sizeLegacyNoScalars(t, i);
       int64_t stride = THTensor_strideLegacyNoScalars(t, i);
       maxOffset_ += (size - 1) * stride;
       sizes_[i] = (IndexT) size;
       strides_[i] = (IndexT) stride;
     }
     if (maxOffset_ > std::numeric_limits<IndexT>::max()) {
       THError("THSyclTensor sizes too large for THSYCLDeviceTensor conversion");
     }
   }
  IndexT stride(int64_t d) {
    return strides_[d];
  }
  IndexT size(int64_t d) {
    return sizes_[d];
  }
  int dim() {
    return dim_;
  }
  template <typename T>
  typename cl::sycl::global_ptr<T>::pointer_t data() {
    return syclAcc_.template get_pointer<T>();
  }

 private:
  ptrdiff_t maxOffset_ = 0;
  const int dim_;
  IndexT sizes_[Dim];
  IndexT strides_[Dim];
  c10::sycl::SYCLAccessor<AccMode> syclAcc_;
};


#endif
