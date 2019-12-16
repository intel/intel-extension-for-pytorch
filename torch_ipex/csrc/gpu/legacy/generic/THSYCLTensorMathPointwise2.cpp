#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathPointwise2.cpp"
#else

void THSYCLTensor_(cmax)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src1, THSYCLTensor *src2) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THSYCLTensor_(nElement)(state, src1) ==
             THSYCLTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src2), TensorMaxOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self, src1);
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src1), THTensor_wrap(src2), TensorMaxOp<scalar_t>());
  }
}


void THSYCLTensor_(cmin)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src1, THSYCLTensor *src2) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THSYCLTensor_(nElement)(state, src1) ==
             THSYCLTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src2), TensorMinOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self, src1);
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src1), THTensor_wrap(src2), TensorMinOp<scalar_t>());
  }
}

#if !defined(THSYCL_REAL_IS_BOOL)

void THSYCLTensor_(addcmul)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *t, scalar_t value, THSYCLTensor *src1, THSYCLTensor *src2) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 4, self_, t, src1, src2));
  if(self_ != t) {
    THSYCLTensor_(resizeAs)(state, self_, t);
    THSYCLTensor_(copy)(state, self_, t);
  } else {
    THArgCheck(THSYCLTensor_(nElement)(state, self_) == THSYCLTensor_(nElement)(state, src1),
               1, "sizes do not match");
  }

  THArgCheck(THSYCLTensor_(nElement)(state, src1) == THSYCLTensor_(nElement)(state, src2),
             3, "sizes do not match");
  at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src1), THTensor_wrap(src2), TensorAddCMulOp<scalar_t>(value));
}

void THSYCLTensor_(addcdiv)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *t, scalar_t value, THSYCLTensor *src1, THSYCLTensor *src2) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 4, self_, t, src1, src2));
  if(self_ != t) {
    THSYCLTensor_(resizeAs)(state, self_, t);
    THSYCLTensor_(copy)(state, self_, t);
  } else {
    THArgCheck(THSYCLTensor_(nElement)(state, self_) == THSYCLTensor_(nElement)(state, src1),
               1, "sizes do not match");
  }

  THArgCheck(THSYCLTensor_(nElement)(state, src1) == THSYCLTensor_(nElement)(state, src2),
             3, "sizes do not match");
  at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src1), THTensor_wrap(src2), TensorAddCDivOp<scalar_t>(value));
}

void THSYCLTensor_(cremainder)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src1, THSYCLTensor *src2) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THSYCLTensor_(nElement)(state, src1) ==
             THSYCLTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src2), TensorCRemainderOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self, src1);
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src1), THTensor_wrap(src2), TensorCRemainderOp<scalar_t>());
  }
}

void THSYCLTensor_(cfmod)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src1, THSYCLTensor *src2) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THSYCLTensor_(nElement)(state, src1) ==
             THSYCLTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src2), TensorCFmodOp<scalar_t>());
  } else {
    THSYCLTensor_(resizeAs)(state, self, src1);
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(self), THTensor_wrap(src1), THTensor_wrap(src2), TensorCFmodOp<scalar_t>());
  }
}

#endif

#endif
