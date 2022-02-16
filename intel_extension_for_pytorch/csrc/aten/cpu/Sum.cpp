#include <ATen/ATen.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>

#include "Sum.h"
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(sum_kernel_stub);

inline c10::ScalarType get_dtype_from_self(
    const at::Tensor& self,
    const c10::optional<c10::ScalarType>& dtype,
    bool promote_integers) {
  if (dtype.has_value()) {
    return dtype.value();
  }
  c10::ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

static c10::ScalarType infer_dtype_from_optional(
    const at::Tensor& self,
    c10::IntArrayRef dim,
    bool keepdim,
    const c10::optional<c10::ScalarType>& opt_dtype,
    const at::Tensor& result) {
  // 'opt_dtype' has the priority for both cases.
  if (result.defined()) {
    // Otherwise, get the result type, if defined.
    return opt_dtype.value_or(result.scalar_type());
  } else {
    // Last case is to get the self type.
    // If the self type is an integer, we promote it to kLong.
    return get_dtype_from_self(self, opt_dtype, true);
  }
}

at::Tensor sum_out_cpu(
    const at::Tensor& input,
    c10::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  auto input_ = input.is_contiguous() ? input : input.contiguous();
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::sum_out_cpu\n");
#endif
  IPEX_RECORD_FUNCTION("torch_ipex::sum_out_cpu", std::vector<c10::IValue>({}));

  at::Tensor output;
  auto out_dtype =
      infer_dtype_from_optional(input_, dim, keepdim, dtype, output);
  at::DimVector dims_(dim);
  at::maybe_wrap_dims(dims_, input_.dim());
  auto shape = at::meta::get_reduction_shape(input_, dims_, keepdim);
  /* resize output */
  output = at::empty(shape, input_.options().dtype(out_dtype));

  auto iter = at::meta::make_reduction_from_out_ty(
      input_, output, dim, keepdim, output.scalar_type());
  if (iter.numel() == 0) {
    output.zero_();
  } else {
#if defined(DYN_DISP_BUILD)
    sum_kernel_stub(kCPU, iter);
#else
    sum_kernel_impl(iter);
#endif
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
