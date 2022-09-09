#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>
#include <torch/all.h>

namespace torch_ipex {
namespace cpu {

class IPEXROIAlignOp : public torch::autograd::Function<IPEXROIAlignOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(
      const at::Tensor& input,
      const at::Tensor& rois,
      double spatial_scale,
      int64_t pooled_height,
      int64_t pooled_width,
      int64_t sampling_ratio,
      bool aligned);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& rois,
      double spatial_scale,
      int64_t pooled_height,
      int64_t pooled_width,
      int64_t sampling_ratio,
      bool aligned);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

at::Tensor ROIAlign_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned);

namespace {

template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T, typename ACC_T>
inline void roi_align_single_framework_forward(
    const T* input,
    const ACC_T count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<ACC_T>>& pre_calc,
    T* output);

template <typename T, typename ACC_T>
inline void roi_align_single_framework_channels_last_forward(
    const T* input,
    const ACC_T count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<ACC_T>>& pre_calc,
    T* output);

template <>
inline void roi_align_single_framework_channels_last_forward<
    at::BFloat16,
    float>(
    const at::BFloat16* input,
    const float count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<float>>& pre_calc,
    at::BFloat16* output);

template <class T>
inline void add(T* address, const T& val);

template <typename T, typename ACC_T>
inline void roi_align_single_framework_backward(
    const T* grad_output,
    const ACC_T count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<ACC_T>>& pre_calc,
    T* grad_input);

template <typename T, typename ACC_T>
inline void roi_align_single_framework_channels_last_backward(
    const T* grad_output,
    const ACC_T count,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    const std::vector<PreCalc<ACC_T>>& pre_calc,
    T* grad_input);

template <typename T, typename ACC_T>
void roi_align_forward_kernel_body(
    int n_rois,
    const T* input,
    const ACC_T& spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const ACC_T* rois,
    T* output,
    bool is_channels_last);

template <typename T, typename ACC_T>
void roi_align_backward_kernel_body(
    int n_rois,
    const T* grad_output,
    const ACC_T& spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    T* grad_input,
    const ACC_T* rois,
    bool is_channels_last);

at::Tensor roi_align_forward_kernel_impl(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned);

at::Tensor roi_align_backward_kernel_impl(
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
    bool is_channels_last);

} // namespace

using roi_align_forward_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    double,
    int64_t,
    int64_t,
    int64_t,
    bool);
DECLARE_DISPATCH(roi_align_forward_kernel_fn, roi_align_forward_kernel_stub);

using roi_align_backward_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    double,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    bool,
    bool);
DECLARE_DISPATCH(roi_align_backward_kernel_fn, roi_align_backward_kernel_stub);

} // namespace cpu
} // namespace torch_ipex