#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/library.h>

#include <exception>
#include <iostream>

namespace at {
namespace autocast {

TORCH_LIBRARY_IMPL(aten, AutocastXPU, m) {
  // fp32
  KERNEL_XPU(binary_cross_entropy, fp32)
  KERNEL_XPU(cross_entropy_loss, fp32)
  KERNEL_XPU(nll_loss_nd, fp32)
  KERNEL_XPU(fft_ifft, fp32)
  KERNEL_XPU(fft_ifft, fp32)
  KERNEL_XPU(fft_fft2, fp32)
  KERNEL_XPU(fft_ifft2, fp32)
  KERNEL_XPU(fft_fftn, fp32)
  KERNEL_XPU(fft_ifftn, fp32)
  KERNEL_XPU(fft_rfft, fp32)
  KERNEL_XPU(fft_irfft, fp32)
  KERNEL_XPU(fft_rfft2, fp32)
  KERNEL_XPU(fft_irfft2, fp32)
  KERNEL_XPU(fft_rfftn, fp32)
  KERNEL_XPU(fft_irfftn, fp32)
  KERNEL_XPU(fft_hfft, fp32)
  KERNEL_XPU(fft_ihfft, fp32)

  //  promote
  KERNEL_XPU(stack, promote)
}
} // namespace autocast
} // namespace at
