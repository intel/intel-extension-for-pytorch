#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorTransformations.h>
#include <ATen/native/cpu/utils.h>
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
#include <ATen/native/xnnpack/Engine.h>
#endif
#include <c10/util/Exception.h>

#include <algorithm>
#include <vector>
#include "ChannelShuffle.h"

#include <torch/extension.h>
#include "csrc/autocast/autocast_mode.h"
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(channel_shuffle_kernel_stub);

at::Tensor channel_shuffle(const at::Tensor& self, int64_t groups) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::channel_shuffle\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::channel_shuffle", std::vector<c10::IValue>({}));

  TORCH_CHECK(
      self.dim() > 2,
      "channel_shuffle expects input with > 2 dims, but got input with sizes ",
      self.sizes());
  int64_t b = self.size(0);
  int64_t c = self.size(1);
  TORCH_CHECK(
      groups > 0,
      "Number of groups to divide channels in must be positive.",
      " Value of groups:",
      groups);
  TORCH_CHECK(
      (c % groups) == 0,
      "Number of channels must be divisible by groups. Got ",
      c,
      " channels and ",
      groups,
      " groups.");

#if defined(C10_MOBILE) && defined(USE_XNNPACK)
  if (self.is_contiguous(MemoryFormat::ChannelsLast) &&
      xnnpack::use_channel_shuffle(self, groups)) {
    return xnnpack::channel_shuffle(self, groups);
  }
#endif

  // custom path for CPU on both Contiuous and ChannelsLast memory format
  if (self.device().type() == c10::DeviceType::CPU &&
      (self.scalar_type() == at::kFloat || self.scalar_type() == at::kDouble)) {
    auto memory_format = self.suggest_memory_format();
    auto output = at::empty({0}, self.options());
    output.resize_(self.sizes(), memory_format);
    auto input = self.contiguous(memory_format);

#if defined(DYN_DISP_BUILD)
    channel_shuffle_kernel_stub(kCPU, output, input, groups);
#else
    channel_shuffle_kernel_impl(output, input, groups);
#endif

    return at::namedinference::propagate_names_if_nonempty(
        output, self.names());
  }

  int64_t oc = c / groups;

  auto input_reshaped = self.view({b, groups, oc, -1});
  // TODO: contiguous can be made to preserve the memory format
  // of the input. However since the above reshape clobbers h and w
  // it may not be safe to do that, since channels_last contiguous
  // may think oc and and the last dim correspond to h,w?
  // It is not clear, however from initial looking around it feels that
  // this may not be correct.
  // In this case channels last will likely require custom implementation
  // if we want to preseve the memory order.
  // XNNPACK has channel shuffle op for NHWC. For mobile usecase this is good.
  // For server we will have to do a custom implementation.
  // For ChannelsFirst, a.k.a Contiguous, memory format we will also need
  // a fast custom implementation perhaps.
  at::Tensor output_tensor =
      input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3})
          .contiguous()
          .reshape(self.sizes());
  return at::namedinference::propagate_names_if_nonempty(
      output_tensor,
      self.has_names() ? self.names() : at::ArrayRef<at::Dimname>{});
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::channel_shuffle"),
      TORCH_FN((&torch_ipex::cpu::channel_shuffle)));
}

} // namespace cpu
} // namespace torch_ipex