#ifndef IPEX_TYPE_DPCPP_CUSTOMIZED_H
#define IPEX_TYPE_DPCPP_CUSTOMIZED_H

#include <ATen/ATen.h>
#include <tensor/Context.h>

namespace at {
namespace AtenIpexTypeDPCPP {

struct DPCPPTensorContext;

at::Tensor convolution_sum(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, at::Tensor& accumu, at::Scalar scale=1.0, at::Scalar alpha=0.f, at::Scalar beta=0.f);

at::Tensor convolution_sum_relu(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, at::Tensor& accumu, at::Scalar scale=1.0, at::Scalar alpha=0.f, at::Scalar beta=0.f);

at::Tensor convolution_relu(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, at::Scalar scale=1.0, at::Scalar alpha=0.f, at::Scalar beta=0.f);

at::Tensor convolution_sigmoid(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, at::Scalar scale=1.0, at::Scalar alpha=0.f, at::Scalar beta=0.f);

at::Tensor & fill_slice_with_index(at::Tensor & t, int dim);

at::Tensor & std_var_out(at::Tensor & result, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt);

std::tuple<Tensor&,Tensor&> std_var_mean_out(const char* fname, Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt);

at::Tensor linear_relu(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias);

at::Tensor mul_add(const Tensor& self, const Tensor& other, const Tensor& accumu, Scalar alpha);

at::Tensor empty_opaque_tensor(DPCPPTensorContext::Meta meta, const TensorOptions& options, c10::optional<MemoryFormat> optional_memory_format);

at::Tensor empty_opaque_qtensor(DPCPPTensorContext::Meta meta, c10::optional<MemoryFormat> optional_memory_format, QuantizerPtr quantizer);

at::Tensor to_plain_if_needed(const Tensor& tensor);

at::Tensor to_plain_if_needed_(const Tensor& tensor);

std::vector<at::Tensor> to_plain_if_needed(TensorList tensor);

at::Tensor new_qtensor(IntArrayRef sizes, const TensorOptions& options, QuantizerPtr quantizer);

at::Tensor q_conv2d_sum_relu(at::Tensor& accumu, const at::Tensor& input, const at::Tensor& packed_weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, double conv_scale, int64_t conv_zero_point, double sum_scale, int64_t sum_zero_point);

at::Tensor quantize_tensor_per_tensor_affine(at::Tensor& qtensor, const at::Tensor& rtensor, double scale, int64_t zero_point);

at::Tensor quantize_tensor_per_channel_affine(at::Tensor& qtensor, const at::Tensor& rtensor, const at::Tensor& scales, const at::Tensor& zero_points, int64_t axis);

at::Tensor dequantize_tensor_per_tensor_affine(at::Tensor& rtensor, const at::Tensor& qtensor, double scale, int64_t zero_point);

at::Tensor dequantize_tensor_per_channel_affine(at::Tensor& rtensor, const at::Tensor& qtensor, const at::Tensor& scales, const at::Tensor& zero_points, int64_t axis);

}
}

#endif
