#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy_nn/generic/MSECriterion.c"
#else
#include <core/SYCLContext.h>
#include <legacy_nn/common.h>
#include <legacy/THSYCLNumerics.h>
#include <legacy/THSYCLDeviceUtils.h>


THSYCL_API void THNN_(MSECriterion_updateOutput)(
          THSYCLState *state,
          THSYCLTensor *input,
          THSYCLTensor *target,
          THSYCLTensor *output,
          int64_t reduction)
{
  THSYCLNN_CHECK_SHAPE(state, input, target);

  if (reduction != at::Reduction::None) {
    THSYCLTensor_(resize0d)(state, output);

    accreal sum = 0;
    int64_t size = THSYCLTensor_(nElement)(state, input);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(input), THTensor_wrap(target), TensorSubOp<scalar_t>());

    THSYCLTensor_(resize1d)(state, input, size);
    THSYCLTensor_(resize1d)(state, target, size);

    dnnl_vec_inner_product_forward((int)size, input, target, output);

    if (reduction == at::Reduction::Mean) {
      sum = THSYCLTensor_(get0d)(state, output);
      sum /= size;
      THSYCLTensor_(set0d)(state, output, (scalar_t)sum);
    }
    return;
  }

  THSYCLTensor_(resizeAs)(state, output, input);
  at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(output), THTensor_wrap(input), THTensor_wrap(target), TensorMSEOp<scalar_t>());
}



THSYCL_API void THNN_(MSECriterion_updateGradInput)(
          THSYCLState *state,
          THSYCLTensor *input,
          THSYCLTensor *target,
          THSYCLTensor *gradOutput,
          THSYCLTensor *gradInput,
          int64_t reduction)
{}

#endif
