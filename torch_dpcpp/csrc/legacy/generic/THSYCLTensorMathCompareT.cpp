#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMathCompareT.cpp"
#else

void THSYCLTensor_(ltTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorLTOp<scalar_t,
                                   bool>());
}

void THSYCLTensor_(gtTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorGTOp<scalar_t,
                                   bool>());
}

void THSYCLTensor_(leTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorLEOp<scalar_t,
                                   bool>());
}

void THSYCLTensor_(geTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));

  THSYCL_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorGEOp<scalar_t,
                                   bool>());
}

void THSYCLTensor_(eqTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorEQOp<scalar_t,
                                   bool>());
}

void THSYCLTensor_(neTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorNEOp<scalar_t,
                                   bool>());
}

void THSYCLTensor_(ltTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorLTOp<scalar_t,
                                scalar_t>());
}

void THSYCLTensor_(gtTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorGTOp<scalar_t,
                                scalar_t>());
}

void THSYCLTensor_(leTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorLEOp<scalar_t,
                                scalar_t>());
}

void THSYCLTensor_(geTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorGEOp<scalar_t,
                                scalar_t>());
}

void THSYCLTensor_(eqTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorEQOp<scalar_t,
                                scalar_t>());
}

void THSYCLTensor_(neTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorNEOp<scalar_t,
                                scalar_t>());
}

void THSYCLTensor_(ltTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorLTOp<scalar_t,
                                             unsigned char>());
}

void THSYCLTensor_(gtTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorGTOp<scalar_t,
                                             unsigned char>());
}

void THSYCLTensor_(leTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorLEOp<scalar_t,
                                             unsigned char>());
}

void THSYCLTensor_(geTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorGEOp<scalar_t,
                                             unsigned char>());
}

void THSYCLTensor_(eqTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorEQOp<scalar_t,
                                             unsigned char>());
}

void THSYCLTensor_(neTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, self_, src1, src2));
  THSYCL_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorNEOp<scalar_t,
                                             unsigned char>());
}



#endif
