#include <ATen/ATen.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>

#include "Sum.h"
#include "utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(sum_kernel_stub);

at::Tensor sum_out_cpu(
    const at::Tensor& input,
    c10::OptionalIntArrayRef opt_dims,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::sum_out_cpu\n");
#endif
  RECORD_FUNCTION("torch_ipex::sum_out_cpu", c10::ArrayRef<c10::IValue>({}));

  // if dtype does not have value, we assign the input type to dtype
  // if dtype is an integer, we promote it to kLong.
  if (!dtype.has_value()) {
    dtype = input.scalar_type();
    if (at::isIntegralType(dtype.value(), /*includeBool=*/true)) {
      dtype = kLong;
    }
  }

  at::DimVector dims_ = at::native::make_dim_vector(opt_dims, input.dim());
  at::maybe_wrap_dims(dims_, input.dim());
  auto shape = at::meta::get_reduction_shape(input, dims_, keepdim);
  at::Tensor output = at::empty(shape, input.options().dtype(dtype));

  auto iter = at::meta::make_reduction_from_out_ty(
      input, output, dims_, keepdim, output.scalar_type());

  // This is a workaround for poor performance on some non-contiguous shapes.
  // These non-contiguous shapes cannot go through vectorized path directly,
  // but can enter the existing vectorized path after converting to contiguous
  // and the contiguous overhead is less than the benefits of vectorized
  // computation. In the future, we need to break the assumption that the output
  // of reduce_sum is always contiguous to improve the sum kernel for these
  // cases.
  auto indtype = iter.dtype();
  if (c10::isFloatingType(indtype) && !input.is_contiguous()) {
    auto in_shapes = iter.shape();
    auto in_strides = iter.strides(1);
    int typesize = iter.input_base().element_size();
    // We assume it will run on isa supporting AVX512. 64 = 512 / 8
    int vecsize = 64 / typesize;
    // To enter the vectorized path, the strides and shapes should satisfy:
    // (in_strides[0] == typesize && in_shapes[0] >= vecsize) or
    // (in_strides[1] == typesize && in_shapes[1] >= vecsize).
    // If we can find an i(i>=2) that satisfies the condition
    // "in_strides[i] == typesize && in_shapes[i] >= vecsize", it will
    // enter the vectorized path after converting the inputs to contiguous.
    for (int i = 2; i < in_shapes.size(); i++) {
      if (in_strides[i] == typesize && in_shapes[i] >= vecsize) {
        auto input_ = input.contiguous();
        return at::sum(input_, dims_, keepdim, dtype);
      }
    }
  }

  if (iter.numel() == 0) {
    output.zero_();
  } else {
    // pointer to sum_kernel_impl(iter);
    sum_kernel_stub(kCPU, iter);
  }

  return output;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::sum.dim_IntList"),
      TORCH_FN((&torch_ipex::cpu::sum_out_cpu)));
}

} // namespace cpu
} // namespace torch_ipex
