#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

namespace torch_ipex {
namespace cpu {

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

template <typename T>
void pre_calc_for_bilinear_interpolate(
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc);

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

template <typename T, typename ACC_T>
void roi_align_forward_kernel_impl(
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
void roi_align_backward_kernel_impl(
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

at::Tensor roi_align_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned);

at::Tensor roi_align_backward_kernel(
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
    bool aligned);

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

} // namespace cpu
} // namespace torch_ipex