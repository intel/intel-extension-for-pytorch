#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <torch/types.h>
#include "csrc/autocast/autocast_mode.h"
#include "csrc/utils/library.h"

#include "TorchVisionNms.h"
#include "csrc/utils/ipex_op_profile.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(nms_kernel_stub);

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::nms\n");
#endif
  IPEX_RECORD_FUNCTION("torch_ipex::nms", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  // pointer to nms_kernel_impl(dets, scores, iou_threshold);
  return nms_kernel_stub(kCPU, dets, scores, iou_threshold);
}

IPEX_TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::nms"),
      TORCH_FN((&torch_ipex::cpu::nms_kernel)));
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::nms", "")
                       .typed<at::Tensor(
                           const at::Tensor& dets,
                           const at::Tensor& scores,
                           double iou_threshold)>();
  return op.call(
      cpu_cached_cast(at::kFloat, dets),
      cpu_cached_cast(at::kFloat, scores),
      iou_threshold);
}

IPEX_TORCH_LIBRARY_IMPL(torchvision, AutocastCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::nms"),
      TORCH_FN((&torch_ipex::autocast::nms_autocast)));
}

} // namespace autocast
} // namespace torch_ipex
