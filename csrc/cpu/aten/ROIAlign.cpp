// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ROIAlign.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/library.h>
#include "autocast/autocast_mode.h"
#include "utils/library.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(roi_align_forward_kernel_stub);
IPEX_DEFINE_DISPATCH(roi_align_backward_kernel_stub);

at::Tensor ROIAlign_forward_impl(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::ROIAlign_forward\n");
#endif
  RECORD_FUNCTION(
      "torch_ipex::ROIAlign_forward", c10::ArrayRef<c10::IValue>({}));

  return roi_align_forward_kernel_stub(
      kCPU,
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

at::Tensor ROIAlign_backward_impl(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned,
    bool is_channels_last) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::ROIAlign_backward\n");
#endif
  RECORD_FUNCTION(
      "torch_ipex::ROIAlign_backward", c10::ArrayRef<c10::IValue>({}));

  return roi_align_backward_kernel_stub(
      kCPU,
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio,
      aligned,
      is_channels_last);
}

at::Tensor ROIAlign_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    c10::SymInt batch_size,
    c10::SymInt channels,
    c10::SymInt height,
    c10::SymInt width,
    int64_t sampling_ratio,
    bool aligned,
    bool is_channels_last) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ROIAlign_backward", "")
                       .typed<decltype(ROIAlign_backward)>();
  return op.call(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio,
      aligned,
      is_channels_last);
}

at::Tensor IPEXROIAlignOp::_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  at::AutoDispatchBelowADInplaceOrView g;
  RECORD_FUNCTION("IPEXROIAlignOp::_forward", c10::ArrayRef<c10::IValue>({}));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ROIAlign_forward", "")
                       .typed<decltype(ROIAlign_forward)>();

  return op.call(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

at::Tensor IPEXROIAlignOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  RECORD_FUNCTION("IPEXROIAlignOp::forward", c10::ArrayRef<c10::IValue>({}));

  ctx->saved_data["input_shape"] = input.sym_sizes();
  ctx->saved_data["spatial_scale"] = spatial_scale;
  ctx->saved_data["pooled_height"] = pooled_height;
  ctx->saved_data["pooled_width"] = pooled_width;
  ctx->saved_data["sampling_ratio"] = sampling_ratio;
  ctx->saved_data["aligned"] = aligned;
  ctx->saved_data["is_channels_last"] =
      input.is_contiguous(at::MemoryFormat::ChannelsLast);
  ctx->save_for_backward({rois});

  return _forward(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

torch::autograd::variable_list IPEXROIAlignOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  RECORD_FUNCTION("IPEXROIAlignOp::backward", c10::ArrayRef<c10::IValue>({}));

  auto input_shape = ctx->saved_data["input_shape"].toSymIntVector();
  auto spatial_scale = ctx->saved_data["spatial_scale"].toDouble();
  auto pooled_height = ctx->saved_data["pooled_height"].toSymInt();
  auto pooled_width = ctx->saved_data["pooled_width"].toSymInt();
  auto sampling_ratio = ctx->saved_data["sampling_ratio"].toInt();
  auto aligned = ctx->saved_data["aligned"].toBool();
  auto is_channels_last = ctx->saved_data["is_channels_last"].toBool();
  auto saved = ctx->get_saved_variables();
  at::Tensor rois = saved[0];

  auto grad_input = ROIAlign_backward(
      grad_outputs[0],
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      input_shape[0],
      input_shape[1],
      input_shape[2],
      input_shape[3],
      sampling_ratio,
      aligned,
      is_channels_last);

  return {
      grad_input,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

at::Tensor ROIAlign_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  if (at::GradMode::is_enabled()) {
    return IPEXROIAlignOp::apply(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned);
  }
  return IPEXROIAlignOp::_forward(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

at::Tensor ROIAlign_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ROIAlign_forward", "")
                       .typed<decltype(torch_ipex::cpu::ROIAlign_forward)>();
  if (input.scalar_type() == at::ScalarType::BFloat16) {
    return op.call(
        input,
        cpu_cached_cast(at::kFloat, rois),
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned);
  } else {
    return op.call(
        input,
        cpu_cached_cast(input.scalar_type(), rois),
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned);
  }
}

} // namespace autocast
} // namespace torch_ipex

namespace {

IPEX_TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "ROIAlign_forward(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, int sampling_ratio, bool aligned) -> Tensor");
  m.impl(
      "ROIAlign_forward",
      c10::DispatchKey::AutogradCPU,
      torch_ipex::cpu::ROIAlign_forward);
  m.impl(
      "ROIAlign_forward",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::ROIAlign_forward);
  m.impl(
      "ROIAlign_forward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::ROIAlign_forward_impl);
  // bw
  m.def(
      "ROIAlign_backward(Tensor grad, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, SymInt batch_size, SymInt channels, SymInt height, SymInt width, int sampling_ratio, bool aligned, bool is_channels_last) -> Tensor");
  m.impl(
      "ROIAlign_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::ROIAlign_backward_impl);
}

IPEX_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.impl(
      "roi_align",
      c10::DispatchKey::AutogradCPU,
      torch_ipex::cpu::ROIAlign_forward);
  m.impl(
      "roi_align",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::ROIAlign_forward);
}

} // namespace
