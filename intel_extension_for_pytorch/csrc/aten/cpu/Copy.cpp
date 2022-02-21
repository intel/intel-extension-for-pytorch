#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/native/quantized/Copy.h>
#include <ATen/native/quantized/affine_quantizer.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/record_function.h>
#include <c10/util/TypeCast.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
#endif

#include "Copy.h"

#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(copy_kernel_stub);

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::copy_\n");
#endif
  IPEX_RECORD_FUNCTION("torch_ipex::copy_", std::vector<c10::IValue>({}));

  auto maybe_outnames =
      at::namedinference::compute_broadcast_outnames(self, src);
  {
    at::NoNamesGuard guard;

#if defined(DYN_DISP_BUILD)
    copy_kernel_stub(kCPU, self, src, non_blocking);
#else
    copy_kernel_impl(self, src, non_blocking);
#endif
  }
  at::namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::copy_"), TORCH_FN((&torch_ipex::cpu::copy_)));
}

} // namespace cpu
} // namespace torch_ipex
