#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathPointwise3.cpp"
#else

#if !defined(THSYCL_REAL_IS_BOOL)

void THSYCLTensor_(pow)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value) {
 THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
 if (self_ == src) {
   if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(1))) {
     at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorPowOp<scalar_t, 1>(value));
   } else if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(2))) {
     at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorPowOp<scalar_t, 2>(value));
   } else if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(3))) {
     at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorPowOp<scalar_t, 3>(value));
#if defined(THSYCL_REAL_IS_HALF) || defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE)
   } else if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-1))) {
     at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorPowOp<scalar_t, -1>(value));
   } else if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-2))) {
     at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorPowOp<scalar_t, -2>(value));
#endif
   } else {
     // fallback implementation using pow
     at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorPowOp<scalar_t, -3>(value));
   }

 } else {
   THSYCLTensor_(resizeAs)(state, self_, src);
   if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(1))) {
     at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorPowOp<scalar_t, 1>(value));
   } else if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(2))) {
     at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorPowOp<scalar_t, 2>(value));
   } else if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(3))) {
     at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorPowOp<scalar_t, 3>(value));
#if defined(THSYCL_REAL_IS_HALF) || defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE)
   } else if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-1))) {
     at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorPowOp<scalar_t, -1>(value));
   } else if (THSYCLNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-2))) {
     at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorPowOp<scalar_t, -2>(value));
#endif
   } else {
     // fallback implementation using pow
     at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorPowOp<scalar_t, -3>(value));
   }
 }
}

#endif

#endif
