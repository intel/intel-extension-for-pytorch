#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

namespace {

at::Tensor nms_kernel_impl(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

}

using nms_kernel_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, double);
DECLARE_DISPATCH(nms_kernel_fn, nms_kernel_stub);

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

} // namespace autocast
} // namespace torch_ipex