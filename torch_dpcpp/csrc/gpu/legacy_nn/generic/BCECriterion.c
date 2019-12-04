#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDPNN/generic/BCECriterion.c"
#else
#include <ATen/dpcpp/SYCLContext.h>
#include <THDPNN/common.h>
#include <THDP/THSYCLNumerics.h>
#include <THDP/THSYCLDeviceUtils.h>


THSYCL_API void THNN_(BCECriterion_updateOutput)(
    THSYCLState *state,
    THSYCLTensor *input,
    THSYCLTensor *target,
    THSYCLTensor *output,
    int64_t reduction,
    THSYCLTensor *weights)
{
  THSYCLNN_CHECK_NELEMENT(state, input, target);
  THSYCLNN_CHECK_NELEMENT(state, input, weights);

  if (reduction == Reduction::None) {
    THSYCLTensor_(resizeAs)(state, output, input);
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(output), THTensor_wrap(input), THTensor_wrap(target), TensorBCEOp<scalar_t>());
    
  if (weights) {
      THSYCLTensor_(cmul)(state, output, output, weights);
    }
    return;
  }

  THSYCLTensor_(resize0d)(state, output);
  scalar_t sum = 0;
  int64_t size = THSYCLTensor_(nElement)(state, input);
  if (weights) {
    at::sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(THTensor_wrap(input), THTensor_wrap(input), THTensor_wrap(target), TensorBCEOp<scalar_t>());
    dnnl_vec_inner_product_forward((int)size, input, weights, output);
    sum = THSYCLTensor_(get0d)(state, output);
  } else {
    THSYCLTensor *t1 = THSYCLTensor_(new)(state);
    THSYCLTensor *t2 = THSYCLTensor_(new)(state);
    THSYCLTensor_(resizeAs)(state, t1, input);
    THSYCLTensor_(resizeAs)(state, t2, input);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(t1), THTensor_wrap(input), TensorLog1Op<scalar_t>());
    dnnl_vec_inner_product_forward((int)size, t1, target, output);
    sum -= THSYCLTensor_(get0d)(state, output);
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(t1), THTensor_wrap(input), TensorLog2Op<scalar_t>());
    at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(THTensor_wrap(t2), THTensor_wrap(target), TensorSub2Op<scalar_t>());
    dnnl_vec_inner_product_forward((int)size, t1, t2, output);
    sum -= THSYCLTensor_(get0d)(state, output);
    THSYCLTensor_(set0d)(state, output, (scalar_t)sum);
  }

  if (reduction == Reduction::Mean) {
    sum /= size;
    THSYCLTensor_(set0d)(state, output, (scalar_t)sum);
  }
    
}


THSYCL_API void THNN_(BCECriterion_updateGradInput)(
    THSYCLState *state,
    THSYCLTensor *input,
    THSYCLTensor *target,
    THSYCLTensor *gradOutput,
    THSYCLTensor *gradInput,
    int64_t reduction,
    THSYCLTensor *weights)
{}

#endif