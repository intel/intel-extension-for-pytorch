#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathPointwise.cpp"
#else

void THSYCLTensor_(sign)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorSignOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src);

    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorSignOp<scalar_t>());
  }
}

void THSYCLTensor_(cbitand)(THSYCLState* state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2) {
#if defined(THSYCL_REAL_IS_HALF) || defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE)
  return THError("cbitand is only supported for integer type tensors");
#else
  THAssert(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THSYCLTensor_(nElement)(state, src1) ==
             THSYCLTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src2), TensorBitAndOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src1);
    // self = src1 / src2
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src1), THTensor_wrap(src2), TensorBitAndOp<scalar_t>());
  }
#endif
}

void THSYCLTensor_(cbitor)(THSYCLState* state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2) {
#if defined(THSYCL_REAL_IS_HALF) || defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE)
  return THError("cbitor is only supported for integer type tensors");
#else
  THAssert(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THSYCLTensor_(nElement)(state, src1) ==
             THSYCLTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src2), TensorBitOrOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src1);
    // self = src1 / src2
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src1), THTensor_wrap(src2), TensorBitOrOp<scalar_t>());
  }
#endif
}

void THSYCLTensor_(cbitxor)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src1, THSYCLTensor* src2) {
#if defined(THSYCL_REAL_IS_HALF) || defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE)
  return THError("cbitxor is only supported for integer type tensors");
#else
  THAssert(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THSYCLTensor_(nElement)(state, src1) ==
             THSYCLTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src2), TensorBitXorOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src1);
    // self = src1 / src2
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src1), THTensor_wrap(src2), TensorBitXorOp<scalar_t>());
  }
#endif
}

void THSYCLTensor_(cmaxValue)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, scalar_t value) {
  if (self == src) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(
        THTensor_wrap(self), TensorMaxValueOp<scalar_t>(value));
  } else {
    THSYCLTensor_(resizeAs)(state, self, src);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(
        THTensor_wrap(self), THTensor_wrap(src), TensorMaxValueOp<scalar_t>(value));
  }
}

#if !defined(THSYCL_REAL_IS_BOOL)

void THSYCLTensor_(clamp)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t min_value, scalar_t max_value) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorClampOp<scalar_t>(min_value, max_value));
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src), TensorClampOp<scalar_t>(min_value, max_value));
  }
}

void THSYCLTensor_(cmul)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2) {
  auto out = at::Tensor(retainTensorImpl(self_));
  at::mul_out(out, at::Tensor(retainTensorImpl(src1)), at::Tensor(retainTensorImpl(src2)));
}

void THSYCLTensor_(cpow)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2) {
 THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
 THArgCheck(THSYCLTensor_(nElement)(state, src1) ==
            THSYCLTensor_(nElement)(state, src2), 3, "sizes do not match");
 if (self_ == src1) {
   // self = pow(self, src2)
   at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src2), TensorCPowOp<scalar_t>());
 } else {
   THSYCLTensor_(resizeAs)(state, self_, src1);
   // self = pow(src1, src2)
   at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src1), THTensor_wrap(src2), TensorCPowOp<scalar_t>());
 }
}

void THSYCLTensor_(clshift)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src1, THSYCLTensor* src2) {
  AT_ERROR("not implemented THSYCLTensor_clshift\n");
}

void THSYCLTensor_(crshift)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src1, THSYCLTensor* src2) {
  AT_ERROR("not implemented THSYCLTensor_crshift\n");
}

#endif

#endif
