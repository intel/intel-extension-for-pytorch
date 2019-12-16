#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathCompare.cpp"
#else

void THSYCLTensor_(ltValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorLTValueOp<scalar_t,
                                  bool>(value));
}

void THSYCLTensor_(gtValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorGTValueOp<scalar_t,
                                  bool>(value));
}

void THSYCLTensor_(leValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorLEValueOp<scalar_t,
                                  bool>(value));
}

void THSYCLTensor_(geValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorGEValueOp<scalar_t,
                                  bool>(value));
}

void THSYCLTensor_(eqValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorEQValueOp<scalar_t,
                                  bool>(value));
}

void THSYCLTensor_(neValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorNEValueOp<scalar_t,
                                  bool>(value));
}

void THSYCLTensor_(ltValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<scalar_t, scalar_t>(state, self_, src,
                                  TensorLTValueOp<scalar_t,
                                  scalar_t>(value));
}

void THSYCLTensor_(gtValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<scalar_t, scalar_t>(state, self_, src,
                               TensorGTValueOp<scalar_t,
                              scalar_t>(value));
}

void THSYCLTensor_(leValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<scalar_t, scalar_t>(state, self_, src,
                               TensorLEValueOp<scalar_t,
                               scalar_t>(value));
}

void THSYCLTensor_(geValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<scalar_t, scalar_t>(state, self_, src,
                               TensorGEValueOp<scalar_t,
                               scalar_t>(value));
}

void THSYCLTensor_(eqValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<scalar_t, scalar_t>(state, self_, src,
                               TensorEQValueOp<scalar_t,
                               scalar_t>(value));
}

void THSYCLTensor_(neValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<scalar_t, scalar_t>(state, self_, src,
                              TensorNEValueOp<scalar_t,
                              scalar_t>(value));
}

void THSYCLTensor_(ltValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorLTValueOp<scalar_t,
                                  unsigned char>(value));
}

void THSYCLTensor_(gtValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorGTValueOp<scalar_t,
                                  unsigned char>(value));
}

void THSYCLTensor_(leValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorLEValueOp<scalar_t,
                                  unsigned char>(value));
}

void THSYCLTensor_(geValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorGEValueOp<scalar_t,
                                  unsigned char>(value));
}

void THSYCLTensor_(eqValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorEQValueOp<scalar_t,
                                  unsigned char>(value));
}

void THSYCLTensor_(neValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value)
{
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, self_, src));
  THSYCL_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorNEValueOp<scalar_t,
                                  unsigned char>(value));
}

#endif
