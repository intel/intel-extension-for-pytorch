#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathPointwise1.cpp"
#else

void THSYCLTensor_(cminValue)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, scalar_t value) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self, src));

  if (self == src) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self), TensorMinValueOp<scalar_t>(value));
  } else {
    THSYCLTensor_(resizeAs)(state, self, src);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src), TensorMinValueOp<scalar_t>(value));
  }
}

#if !defined(THSYCL_REAL_IS_BOOL)

void THSYCLTensor_(tpow)(THSYCLState *state, THSYCLTensor *self_, scalar_t value, THSYCLTensor *src) {
 THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
 if (self_ == src) {
   at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorTPowOp<scalar_t>(value));
 } else {
   THSYCLTensor_(resizeAs)(state, self_, src);
   at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorTPowOp<scalar_t>(value));
 }
}

#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE) || defined(THSYCL_REAL_IS_HALF)

void THSYCLTensor_(atan2)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *tx, THSYCLTensor *ty) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, tx, ty));
  THArgCheck(THSYCLTensor_(nElement)(state, tx) ==
             THSYCLTensor_(nElement)(state, ty), 3, "sizes do not match");
  THSYCLTensor_(resizeAs)(state, self_, tx);
  at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(tx), THTensor_wrap(ty), TensorATan2Op<scalar_t>());
}

void THSYCLTensor_(sigmoid)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorSigmoidOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorSigmoidOp<scalar_t>());
  }
}

void THSYCLTensor_(digamma)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorDigammaOp<scalar_t,accreal>());
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorDigammaOp<scalar_t, accreal>());
  }
}

void THSYCLTensor_(polygamma)(THSYCLState* state, THSYCLTensor* self_, int64_t n, THSYCLTensor* src) {
  AT_ERROR("not implemented THSYCLTensor_polygamma\n");
}

void THSYCLTensor_(erfinv)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorErfinvOp<scalar_t,accreal>());
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorErfinvOp<scalar_t, accreal>());
  }
}

void THSYCLTensor_(lerp)(THSYCLState *state, THSYCLTensor *result, THSYCLTensor *a, THSYCLTensor *b, scalar_t w) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, result, a, b));
  THArgCheck(THSYCLTensor_(nElement)(state, a) ==
             THSYCLTensor_(nElement)(state, b), 3, "sizes do not match");
  THSYCLTensor_(resizeAs)(state, result, a);
  at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(result), THTensor_wrap(a), THTensor_wrap(b), TensorLerpOp<scalar_t>(w));
}

#endif

#endif

#endif
