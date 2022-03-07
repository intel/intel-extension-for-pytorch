#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MatrixRef.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/record_function.h>
#include <core/TensorImplUtils.h>
#include <dnnl.hpp>
#include <oneDNN/oneDNN.h>
#include <torch/autograd.h>
#include <torch/custom_class.h>
#include "comm/ATDispatch.h"

#include "comm/ParamUtils.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;
using namespace torch::autograd;

namespace at {
namespace AtenIpexTypeXPU {

Tensor channel_shuffle(const Tensor& self, int64_t groups) {
  auto smf = self.suggest_memory_format();
  AT_ASSERTM(
      self.dim() > 2,
      "channel_shuffle expects input with > 2 dims, but got input with sizes ",
      self.sizes());
  int64_t b = self.size(0);
  int64_t c = self.size(1);
  AT_ASSERTM(
      groups > 0,
      "Number of groups to divide channels in must be positive.",
      " Value of groups:",
      groups);
  AT_ASSERTM(
      (c % groups) == 0,
      "Number of channels must be divisible by groups. Got ",
      c,
      " channels and ",
      groups,
      " groups.");

  int64_t oc = c / groups;

  auto input_reshaped = self.view({b, groups, oc, -1});

  Tensor output_tensor =
      input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3})
          .reshape(self.sizes());
  return output_tensor.contiguous(smf);
}

} // namespace AtenIpexTypeXPU
} // namespace at
