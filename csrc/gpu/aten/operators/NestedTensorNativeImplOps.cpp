#include <ATen/NativeFunctions.h>
#include "NestedTensorXPUNativeFunctions.h"
namespace at {
namespace AtenIpexTypeNestedTensorXPU {
::std::tuple<at::Tensor, at::Tensor> native_dropout(
    const at::Tensor& input,
    double p,
    c10::optional<bool> train) {
  return at::native::native_dropout_nested(input, p, train);
}

at::Tensor native_dropout_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double scale) {
  return at::native::native_dropout_backward(grad_output, mask, scale);
}

at::Tensor abs(const at::Tensor& self) {
  return at::native::NestedTensor_abs(self);
}

at::Tensor& abs_(at::Tensor& self) {
  return at::native::NestedTensor_abs_(self);
}

at::Tensor sgn(const at::Tensor& self) {
  return at::native::NestedTensor_sgn(self);
}

at::Tensor& sgn_(at::Tensor& self) {
  return at::native::NestedTensor_sgn_(self);
}

at::Tensor add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  return at::native::NestedTensor_add_Tensor(self, other, alpha);
}

at::Tensor& add_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  return at::native::NestedTensor_add__Tensor(self, other, alpha);
}

at::Tensor logical_not(const at::Tensor& self) {
  return at::native::NestedTensor_logical_not(self);
}

at::Tensor& logical_not_(at::Tensor& self) {
  return at::native::NestedTensor_logical_not_(self);
}

::std::vector<at::Tensor> chunk(
    const at::Tensor& self,
    int64_t chunks,
    int64_t dim) {
  return at::native::chunk_nested_tensor(self, chunks, dim);
}

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  return at::native::copy_nested_(self, src, non_blocking);
}

at::Tensor cos(const at::Tensor& self) {
  return at::native::cos_nested(self);
}

at::Tensor div(const at::Tensor& self, const at::Tensor& other) {
  return at::native::NestedTensor_div_Tensor(self, other);
}

at::Tensor div(const at::Tensor& self, const at::Scalar& other) {
  return at::native::NestedTensor_div_Scalar(self, other);
}

at::Tensor embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  return at::native::NestedTensor_embedding(
      weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

at::Tensor empty_like(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  return at::native::empty_like_nested(
      self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor& fill_(at::Tensor& self, const at::Scalar& value) {
  return at::native::fill_nested_(self, value);
}

at::Tensor& fill_(at::Tensor& self, const at::Tensor& value) {
  return at::native::fill_nested_(self, value);
}

bool is_same_size(const at::Tensor& self, const at::Tensor& other) {
  return at::native::nested_is_same_size(self, other);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps) {
  return at::native::nested_layer_norm(
      input, normalized_shape, weight, bias, eps);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    ::std::array<bool, 3> output_mask) {
  return at::native::layer_norm_backward_nested(
      grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
}

at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  return at::native::nested_linear(input, weight, bias);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
    const at::Tensor& self,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    ::std::array<bool, 3> output_mask) {
  return at::native::nested_linear_backward(
      self, grad_output, weight, output_mask);
}

at::Tensor matmul(const at::Tensor& self, const at::Tensor& other) {
  return at::native::matmul_nested(self, other);
}

at::Tensor& matmul_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  return at::native::matmul_out_nested(self, other, out);
}

::std::tuple<at::Tensor, at::Tensor> matmul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& other,
    ::std::array<bool, 2> mask) {
  return at::native::matmul_backward_nested(grad, self, other, mask);
}

at::Tensor mul(const at::Tensor& self, const at::Tensor& other) {
  return at::native::NestedTensor_mul_Tensor(self, other);
}

at::Tensor& mul_(at::Tensor& self, const at::Tensor& other) {
  return at::native::NestedTensor_mul__Tensor(self, other);
}

at::Tensor mul(const at::Tensor& self, const at::Scalar& other) {
  return at::native::NestedTensor_mul_Scalar(self, other);
}

at::Tensor& mul_(at::Tensor& self, const at::Scalar& other) {
  return at::native::NestedTensor_mul__Scalar(self, other);
}

at::Tensor ones_like(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  return at::native::ones_like(
      self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor neg(const at::Tensor& self) {
  return at::native::NestedTensor_neg(self);
}

at::Tensor& neg_(at::Tensor& self) {
  return at::native::NestedTensor_neg_(self);
}

at::Tensor relu(const at::Tensor& self) {
  return at::native::NestedTensor_relu(self);
}

at::Tensor& relu_(at::Tensor& self) {
  return at::native::NestedTensor_relu_(self);
}

at::Tensor gelu(const at::Tensor& self, c10::string_view approximate) {
  return at::native::NestedTensor_gelu(self, approximate);
}

at::Tensor& gelu_(at::Tensor& self, c10::string_view approximate) {
  return at::native::NestedTensor_gelu_(self, approximate);
}

at::Tensor gelu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate) {
  return at::native::gelu_backwards_nested(grad_output, self, approximate);
}

at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index) {
  return at::native::select_nested(self, dim, index);
}

at::Tensor _nested_select_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    int64_t dim,
    int64_t index) {
  return at::native::_nested_select_backward_symint(
      grad_output, self, dim, index);
}

at::Tensor silu(const at::Tensor& self) {
  return at::native::NestedTensor_silu(self);
}

at::Tensor& silu_(at::Tensor& self) {
  return at::native::NestedTensor_silu_(self);
}

at::Tensor silu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  return at::native::silu_backward_nested(grad_output, self);
}

at::Tensor detach(const at::Tensor& self) {
  return at::native::detach(self);
}

at::Tensor _softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return at::native::softmax_nested(self, dim, half_to_float);
}

at::Tensor _softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {
  return at::native::nested_softmax_backward(
      grad_output, output, dim, input_dtype);
}

::std::vector<at::Tensor> split_with_sizes(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  return at::native::split_with_sizes_nested(self, split_sizes, dim);
}

at::Tensor squeeze(const at::Tensor& self) {
  return at::native::squeeze_nested(self);
}

at::Tensor squeeze(const at::Tensor& self, int64_t dim) {
  return at::native::squeeze_dim_nested(self, dim);
}

at::Tensor squeeze(const at::Tensor& self, at::IntArrayRef dim) {
  return at::native::squeeze_dim_nested(self, dim);
}

at::Tensor tanh(const at::Tensor& self) {
  return at::native::NestedTensor_tanh(self);
}

at::Tensor& tanh_(at::Tensor& self) {
  return at::native::NestedTensor_tanh_(self);
}

at::Tensor threshold_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold) {
  return at::native::threshold_backwards_nested(grad_output, self, threshold);
}

at::Tensor transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  return at::native::transpose_nested(self, dim0, dim1);
}

at::Tensor _nested_tensor_size(const at::Tensor& self) {
  return at::native::_nested_tensor_size(self);
}

at::Tensor _nested_tensor_strides(const at::Tensor& self) {
  return at::native::_nested_tensor_strides(self);
}

at::Tensor _nested_from_padded_and_nested_example(
    const at::Tensor& padded,
    const at::Tensor& nt_example) {
  return at::native::NestedTensor_from_padded_and_nested_example(
      padded, nt_example);
}

at::Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
  return at::native::unsqueeze_nested(self, dim);
}

at::Tensor clone(
    const at::Tensor& self,
    c10::optional<at::MemoryFormat> memory_format) {
  return at::native::clone_nested(self, memory_format);
}

at::Tensor& zero_(at::Tensor& self) {
  return at::native::zero_nested_(self);
}

at::Tensor sub(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  return at::native::NestedTensor_sub_Tensor(self, other, alpha);
}

at::Tensor values(const at::Tensor& self) {
  return at::native::values_nested(self);
}

at::Tensor _to_copy(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<at::MemoryFormat> memory_format) {
  return at::native::_to_copy_nested(
      self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

at::Tensor masked_fill(
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  return at::native::NestedTensor_masked_fill(self, mask, value);
}

at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
  return at::native::view_nested(self, size);
}

at::Tensor& normal_(
    at::Tensor& self,
    double mean,
    double std,
    c10::optional<at::Generator> generator) {
  return at::native::normal_nested_(self, mean, std, generator);
}

} // namespace AtenIpexTypeNestedTensorXPU
} // namespace at