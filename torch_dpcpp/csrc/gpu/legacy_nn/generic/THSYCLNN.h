#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDPNN/generic/THSYCLNN.h"
#else

#include <ATen/core/Reduction.h>

THSYCL_API void THNN_(AbsCriterion_updateOutput)(
          THSYCLState *state,            // library's state
          THSYCLTensor *input,             // input tensor
          THSYCLTensor *target,            // tensor with target values
          THSYCLTensor *output,            // [OUT] a one-element tensor with loss
          int64_t reduction);

THSYCL_API void THNN_(AbsCriterion_updateGradInput)(
          THSYCLState *state,
          THSYCLTensor *input,
          THSYCLTensor *target,
          THSYCLTensor *gradOutput,
          THSYCLTensor *gradInput,
          int64_t reduction);

THSYCL_API void THNN_(ClassNLLCriterion_updateOutput)(
                  THSYCLState *state,
                  THSYCLTensor *input,
                  THSYCLIndexTensor *target,
                  THSYCLTensor *output,
                  int64_t reduction,
                  THSYCLTensor *weights,       // [OPTIONAL]
                  THSYCLTensor *total_weight,
                  int64_t ignore_index);

THSYCL_API void THNN_(ClassNLLCriterion_updateGradInput)(
                  THSYCLState *state,
                  THSYCLTensor *input,
                  THSYCLIndexTensor *target,
                  THSYCLTensor *gradOutput,
                  THSYCLTensor *gradInput,
                  int64_t reduction,
                  THSYCLTensor *weights,       // [OPTIONAL]
                  THSYCLTensor *total_weight,
                  int64_t ignore_index);

THSYCL_API void THNN_(MSECriterion_updateOutput)(
          THSYCLState *state,
          THSYCLTensor *input,
          THSYCLTensor *target,
          THSYCLTensor *output,
          int64_t reduction);

THSYCL_API void THNN_(MSECriterion_updateGradInput)(
          THSYCLState *state,
          THSYCLTensor *input,
          THSYCLTensor *target,
          THSYCLTensor *gradOutput,
          THSYCLTensor *gradInput,
          int64_t reduction);

THSYCL_API void THNN_(BCECriterion_updateOutput)(
    THSYCLState *state,
    THSYCLTensor *input,
    THSYCLTensor *target,
    THSYCLTensor *output,
    int64_t reduction,
    THSYCLTensor *weights);

THSYCL_API void THNN_(BCECriterion_updateGradInput)(
    THSYCLState *state,
    THSYCLTensor *input,
    THSYCLTensor *target,
    THSYCLTensor *gradOutput,
    THSYCLTensor *gradInput,
    int64_t reduction,
    THSYCLTensor *weights);

#endif
