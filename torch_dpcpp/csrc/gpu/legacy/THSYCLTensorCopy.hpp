#pragma once

#include <legacy/THSYCLTensorCopy.h>

template <typename ScalarTypeDst, typename ScalarTypeSrc>
void THSYCL_copyTensor(THSYCLState* state, THSYCLTensor* dst, THSYCLTensor* src);

template <typename ScalarType>
THSYCLTensor *THSYCLTensor_newClone(THSYCLState *state, THSYCLTensor *self);

template <typename ScalarType>
THSYCLTensor *THSYCLTensor_newContiguous(THSYCLState *state, THSYCLTensor *self);

template <typename ScalarType>
void THSYCLTensor_freeCopyTo(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *dst);

template <typename ScalarType>
void THSYCLTensor_copyIgnoringOverlaps(THSYCLState* state, THSYCLTensor* dst, THSYCLTensor* src);

