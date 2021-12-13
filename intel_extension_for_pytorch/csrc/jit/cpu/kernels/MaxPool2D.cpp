#include "MaxPool2D.h"
#include "csrc/aten/cpu/Pooling.h"
#include "csrc/utils/utils.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor dil_max_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_max_pool2d", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(
      std::all_of(
          dilation.cbegin(), dilation.cend(), [](int64_t i) { return 1 == i; }),
      "dil_max_pool2d does not support dilation case");
  return pooling_impl(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      ideep::algorithm::pooling_max);
}

} // namespace cpu
} // namespace torch_ipex
