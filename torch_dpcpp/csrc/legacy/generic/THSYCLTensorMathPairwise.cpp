#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMathPairwise.cpp"
#else

int THSYCLTensor_(equal)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src_) {
  AT_ERROR("not implemented THSYCLTensor_equal\n");
}

void THSYCLTensor_(bitand)(THSYCLState* state, THSYCLTensor *self_, THSYCLTensor *src_, scalar_t value)
{
#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE) || defined(THSYCL_REAL_IS_HALF)
  return THError("bitand only supported for integer type tensors");
#else
  if (self_ == src_) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorBitAndConstantOp<scalar_t>(value));

  } else {
    THSYCLTensor_(resizeAs)(state, self_, src_);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src_), TensorBitAndConstantOp<scalar_t>(value));

  }
#endif
}


void THSYCLTensor_(bitor)(THSYCLState* state, THSYCLTensor *self_, THSYCLTensor *src_, scalar_t value)
{
#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE) || defined(THSYCL_REAL_IS_HALF)
  return THError("bitor only supported for integer type tensors");
#else
  if (self_ == src_) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorBitOrConstantOp<scalar_t>(value));

  } else {
    THSYCLTensor_(resizeAs)(state, self_, src_);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src_), TensorBitOrConstantOp<scalar_t>(value));

  }
#endif
}

void THSYCLTensor_(bitxor)(THSYCLState* state, THSYCLTensor* self_,
    THSYCLTensor* src_, scalar_t value) {
  AT_ERROR("not implemented THSYCLTensor_bitxor\n");
}

#if !defined(THSYCL_REAL_IS_BOOL)

void THSYCLTensor_(lshift)(THSYCLState* state, THSYCLTensor* self_,
    THSYCLTensor* src_, scalar_t value) {
  AT_ERROR("not implemented THSYCLTensor_lshift\n");
}

void THSYCLTensor_(rshift)(THSYCLState* state, THSYCLTensor* self_,
    THSYCLTensor* src_, scalar_t value) {
  AT_ERROR("not implemented THSYCLTensor_rshift\n");
}

void THSYCLTensor_(fmod)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src_, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
     at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorFmodOp<scalar_t>(value));
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src_);
     at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src_), TensorFmodOp<scalar_t>(value));
  }

}

void THSYCLTensor_(remainder)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src_, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    at::sycl::SYCL_tensor_apply1<scalar_t>(THTensor_wrap(self_), TensorRemainderOp<scalar_t>(value));
  } else {
    THSYCLTensor_(resizeAs)(state, self_, src_);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(self_), THTensor_wrap(src_), TensorRemainderOp<scalar_t>(value));
  }

}

void THSYCLTensor_(tril)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src_, int64_t k)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(!src_->is_empty() && src_->dim() == 2, 1, "expected a matrix");

  if (self_ != src_)
    THSYCLTensor_(resizeAs)(state, self_, src_);

  int64_t stride0 = self_->stride(0);
  int64_t stride1 = self_->stride(1);

  TensorTriOp<scalar_t, 0> op(stride0, stride1, k);

  if (self_ == src_) {
    at::sycl::SYCL_tensor_apply1<scalar_t, decltype(op), true>(THTensor_wrap(src_), op);
  } else {
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t, decltype(op), true>(THTensor_wrap(self_), THTensor_wrap(src_), op);
  }
}

void THSYCLTensor_(triu)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src_, int64_t k)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(!src_->is_empty() && src_->dim() == 2, 1, "expected a matrix");

  if (self_ != src_)
    THSYCLTensor_(resizeAs)(state, self_, src_);

  int64_t stride0 = self_->stride(0);
  int64_t stride1 = self_->stride(1);

  TensorTriOp<scalar_t, 1> op(stride0, stride1, k);

  if (self_ == src_) {
    at::sycl::SYCL_tensor_apply1<scalar_t, decltype(op), true>(THTensor_wrap(src_), op);
  } else {
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t, decltype(op), true>(THTensor_wrap(self_), THTensor_wrap(src_), op);
  }
}

#endif

#endif
