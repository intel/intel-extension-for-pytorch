#pragma once

#include <ATen/ATen.h>
#include <csrc/dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

template <typename scalar_t, int out_ndims, int interp_size>
void cpu_upsample_generic(at::TensorIterator& iter);

template <typename scalar_t, typename scale_type>
void cpu_upsample_nearest_channels_last(
    const at::Tensor& output_,
    const at::Tensor& input_,
    const scale_type& scales);

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_channels_last(
    const at::Tensor& output_,
    const at::Tensor& input_,
    bool align_corners,
    const scale_type& scales);

template <int out_ndims, typename scale_type, class F>
void upsample_generic_Nd_kernel_body(
    const at::Tensor& output,
    const at::Tensor& input,
    bool align_corners,
    const scale_type& scales);

template <typename scalar_t, typename scale_type>
void cpu_upsample_nearest_backward(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    const scale_type& scales);

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_backward(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    bool align_corners,
    const scale_type& scales);

/*********************************************/
at::Tensor upsample_nearest1d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales);

at::Tensor upsample_nearest1d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales);

at::Tensor upsample_nearest2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

at::Tensor upsample_nearest2d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

at::Tensor upsample_nearest3d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

at::Tensor upsample_nearest3d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

at::Tensor upsample_linear1d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales);

at::Tensor upsample_linear1d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales);

at::Tensor upsample_bilinear2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

at::Tensor upsample_bilinear2d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

at::Tensor upsample_trilinear3d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

at::Tensor upsample_trilinear3d_backward_out_cpu(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

#if defined(DYN_DISP_BUILD)
namespace {
#endif

void upsample_nearest1d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::optional<double> scales_w);

void upsample_nearest2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_nearest3d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_linear1d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    bool align_corners,
    c10::optional<double> scales_w);

void upsample_bilinear2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_trilinear3d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_bicubic2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_nearest1d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    c10::optional<double> scales_w);

void upsample_nearest2d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_nearest3d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_linear1d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_w);

void upsample_bilinear2d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_trilinear3d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

#if defined(DYN_DISP_BUILD)
}
#endif

using upsample_nearest1d_kernel_fn =
    void (*)(const at::Tensor&, const at::Tensor&, c10::optional<double>);
DECLARE_DISPATCH(upsample_nearest1d_kernel_fn, upsample_nearest1d_kernel_stub);

using upsample_nearest2d_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(upsample_nearest2d_kernel_fn, upsample_nearest2d_kernel_stub);

using upsample_nearest3d_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    c10::optional<double>,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(upsample_nearest3d_kernel_fn, upsample_nearest3d_kernel_stub);

using upsample_linear1d_kernel_fn =
    void (*)(const at::Tensor&, const at::Tensor&, bool, c10::optional<double>);
DECLARE_DISPATCH(upsample_linear1d_kernel_fn, upsample_linear1d_kernel_stub);

using upsample_bilinear2d_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(
    upsample_bilinear2d_kernel_fn,
    upsample_bilinear2d_kernel_stub);

using upsample_trilinear3d_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    c10::optional<double>,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(
    upsample_trilinear3d_kernel_fn,
    upsample_trilinear3d_kernel_stub);

using upsample_bicubic2d_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(upsample_bicubic2d_kernel_fn, upsample_bicubic2d_kernel_stub);

using upsample_nearest1d_backward_kernel_fn =
    void (*)(const at::Tensor&, const at::Tensor&, c10::optional<double>);
DECLARE_DISPATCH(
    upsample_nearest1d_backward_kernel_fn,
    upsample_nearest1d_backward_kernel_stub);

using upsample_nearest2d_backward_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(
    upsample_nearest2d_backward_kernel_fn,
    upsample_nearest2d_backward_kernel_stub);

using upsample_nearest3d_backward_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    c10::optional<double>,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(
    upsample_nearest3d_backward_kernel_fn,
    upsample_nearest3d_backward_kernel_stub);

using upsample_linear1d_backward_kernel_fn =
    void (*)(const at::Tensor&, const at::Tensor&, bool, c10::optional<double>);
DECLARE_DISPATCH(
    upsample_linear1d_backward_kernel_fn,
    upsample_linear1d_backward_kernel_stub);

using upsample_bilinear2d_backward_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(
    upsample_bilinear2d_backward_kernel_fn,
    upsample_bilinear2d_backward_kernel_stub);

using upsample_trilinear3d_backward_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    c10::optional<double>,
    c10::optional<double>,
    c10::optional<double>);
DECLARE_DISPATCH(
    upsample_trilinear3d_backward_kernel_fn,
    upsample_trilinear3d_backward_kernel_stub);

} // namespace cpu
} // namespace torch_ipex