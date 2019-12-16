#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorScatterGather.h"
#else
void THSYCLTensor_(gather)(THSYCLState* state, THSYCLTensor* tensor, THSYCLTensor* src, int dim, THSyclLongTensor* index);

void THSYCLTensor_(scatter)(THSYCLState* state, THSYCLTensor* tensor, int dim, THSyclLongTensor* index, THSYCLTensor* src);

void THSYCLTensor_(scatterAdd)(THSYCLState* state, THSYCLTensor* tensor, int dim, THSyclLongTensor* index, THSYCLTensor* src);

void THSYCLTensor_(scatterFill)(THSYCLState* state, THSYCLTensor* tensor, int dim, THSyclLongTensor* index, scalar_t value);
#endif
