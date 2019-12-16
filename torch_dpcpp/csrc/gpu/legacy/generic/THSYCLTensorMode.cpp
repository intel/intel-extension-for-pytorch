#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMode.cpp"
#else

void THSYCLTensor_(mode)(THSYCLState* state,
                         THSYCLTensor* values,
                         THSyclLongTensor* indices,
                         THSYCLTensor* input,
                         int dimension,
                         int keepdim) {
  AT_ERROR("not implemented THSYCLTensor_mode\n");
}

#endif
