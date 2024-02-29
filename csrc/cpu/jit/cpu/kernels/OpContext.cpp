#include "OpContext.h"
#include <torch/all.h>
#include "ConvPacked.h"
#include "ConvTransposePacked.h"
#include "LinearMKLPacked.h"
#include "LinearPacked.h"
#include "LinearWoqPacked.h"

namespace torch_ipex {
namespace cpu {

template <typename T1, typename T2>
void load_from_ctx_template(T1* self, c10::intrusive_ptr<T2> other) {
  auto& other_ctx_ = other->get_context();
  auto loaded_weight = other_ctx_.at_weight_;
  auto loaded_bias = other_ctx_.at_bias_;
  self->get_context().at_weight_.copy_(loaded_weight);
  if (loaded_bias.has_value()) {
    self->get_context().at_bias_.value().copy_(loaded_bias.value());
  }
  return;
}

template <>
void load_from_ctx_template<IpexLinearMKLOpContext, MKLOpContext>(
    IpexLinearMKLOpContext* self,
    c10::intrusive_ptr<MKLOpContext> other) {
  auto& other_ctx_ = other->get_context();
  auto loaded_weight = other_ctx_.at_weight_;
  auto loaded_bias = other_ctx_.at_bias_;
  self->get_context().at_weight_.copy_(loaded_weight);
  if (loaded_bias.has_value()) {
    self->get_context().at_bias_.value().copy_(loaded_bias.value());
  }
  self->get_context().ori_weight_.copy_(other->get_context().ori_weight_);
  return;
}
c10::intrusive_ptr<ConvolutionOpContext> IpexConvolutionOpContext::
    create_context(
        at::Tensor&& weight,
        c10::optional<at::Tensor>&& bias,
        std::vector<int64_t>&& stride,
        std::vector<int64_t>&& padding,
        std::vector<int64_t>&& dilation,
        int64_t groups,
        bool weight_is_channels_last,
        std::vector<int64_t>&& input_size,
        const ideep::attr_t& attr) {
  auto op_context = torch_ipex::cpu::detail::convolution::create(
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      weight_is_channels_last,
      input_size,
      attr);
  return c10::make_intrusive<IpexConvolutionOpContext>(
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      std::move(input_size),
      std::move(op_context));
}

std::vector<int64_t> ConvolutionOpContext::get_stride() {
  return this->get_context().stride_;
}

std::vector<int64_t> ConvolutionOpContext::get_padding() {
  return this->get_context().padding_;
}

std::vector<int64_t> ConvolutionOpContext::get_dilation() {
  return this->get_context().dilation_;
}

int64_t ConvolutionOpContext::get_groups() {
  return this->get_context().groups_;
}

at::Tensor IpexConvolutionOpContext::run(
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::convolution::run(op_context_, input, attr);
}

at::Tensor& IpexConvolutionOpContext::run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::convolution::run(
      op_context_, input, accumu, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IpexConvolutionOpContext::
    run_backward(
        const at::Tensor& input,
        const at::Tensor& grad_output,
        std::array<bool, 3> output_mask) {
  return torch_ipex::cpu::detail::convolution::run_backward(
      op_context_, input, grad_output, output_mask);
}

at::Tensor IpexConvolutionOpContext::get_at_packed_weight() {
  return torch_ipex::cpu::detail::convolution::get_at_packed_weight(
      op_context_);
}

c10::optional<at::Tensor> IpexConvolutionOpContext::get_at_bias() {
  return torch_ipex::cpu::detail::convolution::get_at_bias(op_context_);
}

at::Tensor IpexConvolutionOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::convolution::pack(op_context_, tensor);
}

at::Tensor IpexConvolutionOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::convolution::unpack(op_context_, tensor);
}

detail::ContextConvolution& IpexConvolutionOpContext::get_context() {
  return op_context_;
}

at::Tensor IpexConvolutionOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

void IpexConvolutionOpContext::load_from_ctx(
    c10::intrusive_ptr<ConvolutionOpContext> other) {
  load_from_ctx_template(this, other);
}

c10::intrusive_ptr<LinearOpContext> IpexLinearOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size) {
  auto op_context =
      torch_ipex::cpu::detail::linear::create(weight, bias, batch_size);
  return c10::make_intrusive<IpexLinearOpContext>(
      batch_size, std::move(op_context));
}

at::Tensor IpexLinearOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

at::Tensor IpexLinearOpContext::run(
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::linear::run(op_context_, input, attr);
}

at::Tensor& IpexLinearOpContext::run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::linear::run(op_context_, input, accumu, attr);
}

at::Tensor IpexLinearOpContext::run_with_binary_post_op(
    const at::Tensor& input,
    const std::vector<ideep::tensor>& post_op_src,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::linear::run(
      op_context_, input, post_op_src, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IpexLinearOpContext::
    run_backward(
        const at::Tensor& input,
        const at::Tensor& grad_output,
        std::array<bool, 3> output_mask) {
  return torch_ipex::cpu::detail::linear::run_backward(
      op_context_, input, grad_output, output_mask);
}

at::Tensor IpexLinearOpContext::get_at_packed_weight() {
  return op_context_.at_weight_;
}

c10::optional<at::Tensor> IpexLinearOpContext::get_at_bias() {
  return op_context_.at_bias_;
}

detail::ContextLinear& IpexLinearOpContext::get_context() {
  return op_context_;
}

at::Tensor IpexLinearOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::linear::pack(op_context_, tensor);
}

at::Tensor IpexLinearOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::linear::unpack(op_context_, tensor);
}

void IpexLinearOpContext::load_from_ctx(
    c10::intrusive_ptr<LinearOpContext> other) {
  load_from_ctx_template(this, other);
}

c10::intrusive_ptr<ConvTransposeOpContext> IpexConvTransposeOpContext::
    create_context(
        at::Tensor&& weight,
        c10::optional<at::Tensor>&& bias,
        std::vector<int64_t>&& stride,
        std::vector<int64_t>&& padding,
        std::vector<int64_t>&& output_padding,
        std::vector<int64_t>&& dilation,
        int64_t groups,
        bool weight_is_channels_last,
        std::vector<int64_t>&& input_size) {
  auto op_context = torch_ipex::cpu::detail::conv_transpose::create(
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      weight_is_channels_last,
      input_size);
  return c10::make_intrusive<IpexConvTransposeOpContext>(
      std::move(stride),
      std::move(padding),
      std::move(output_padding),
      std::move(dilation),
      std::move(input_size),
      std::move(op_context));
}

c10::intrusive_ptr<MKLOpContext> IpexLinearMKLOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size) {
  auto op_context =
      torch_ipex::cpu::detail::mkl_sgemm::create(weight, bias, batch_size);
  return c10::make_intrusive<IpexLinearMKLOpContext>(
      batch_size, std::move(op_context));
}

at::Tensor IpexLinearMKLOpContext::get_at_packed_weight() {
  return op_context_.at_weight_;
}

c10::optional<at::Tensor> IpexLinearMKLOpContext::get_at_bias() {
  return op_context_.at_bias_;
}

at::Tensor IpexLinearMKLOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

at::Tensor IpexLinearMKLOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::mkl_sgemm::pack(op_context_, tensor);
}

at::Tensor IpexLinearMKLOpContext::run(const at::Tensor& input) {
  return torch_ipex::cpu::detail::mkl_sgemm::run(op_context_, input);
}

at::Tensor& IpexLinearMKLOpContext::run(
    const at::Tensor& input,
    at::Tensor& accumu) {
  return torch_ipex::cpu::detail::mkl_sgemm::run(op_context_, input, accumu);
}

at::Tensor IpexLinearMKLOpContext::to_public(const at::Tensor& tensor) {
  return op_context_.ori_weight_.clone();
}

detail::ContextLinearMKL& IpexLinearMKLOpContext::get_context() {
  return op_context_;
}

int64_t IpexLinearMKLOpContext::get_out_features() {
  return op_context_.sgemm_sizes_[2];
}

int64_t IpexLinearMKLOpContext::get_in_features() {
  return op_context_.sgemm_sizes_[1];
}

void IpexLinearMKLOpContext::load_from_ctx(
    c10::intrusive_ptr<MKLOpContext> other) {
  load_from_ctx_template(this, other);
}

at::Tensor IpexConvTransposeOpContext::run(
    const at::Tensor& input,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::conv_transpose::run(op_context_, input, attr);
}

at::Tensor& IpexConvTransposeOpContext::run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr) {
  return torch_ipex::cpu::detail::conv_transpose::run(
      op_context_, input, accumu, attr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IpexConvTransposeOpContext::
    run_backward(
        const at::Tensor& input,
        const at::Tensor& grad_output,
        std::array<bool, 3> output_mask) {
  return torch_ipex::cpu::detail::conv_transpose::run_backward(
      op_context_, input, grad_output, output_mask);
}

at::Tensor IpexConvTransposeOpContext::get_at_packed_weight() {
  return torch_ipex::cpu::detail::conv_transpose::get_at_packed_weight(
      op_context_);
}

c10::optional<at::Tensor> IpexConvTransposeOpContext::get_at_bias() {
  return op_context_.at_bias_;
}

at::Tensor IpexConvTransposeOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::conv_transpose::pack(op_context_, tensor);
}

at::Tensor IpexConvTransposeOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::conv_transpose::unpack(op_context_, tensor);
}

void IpexConvTransposeOpContext::may_repack(std::vector<int64_t> input_size) {
  if (input_size_.empty() || input_size_ != input_size) {
    input_size_ = input_size;
    torch_ipex::cpu::detail::conv_transpose::repack_for(
        op_context_, input_size);
  }
  return;
}

at::Tensor IpexConvTransposeOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

detail::ContextConvTranspose& IpexConvTransposeOpContext::get_context() {
  return op_context_;
}

void IpexConvTransposeOpContext::load_from_ctx(
    c10::intrusive_ptr<ConvTransposeOpContext> other) {
  load_from_ctx_template(this, other);
}

#ifdef USE_LIBXSMM
// For weight-only quantization
c10::intrusive_ptr<WoqLinearOpContext> IpexWoqLinearOpContext::create_context(
    at::Tensor&& weight,
    int64_t weight_dtype, // int8=1, int4=2, nf4=3
    std::vector<int64_t>&& weight_shape,
    at::Tensor&& scales_fp32,
    c10::optional<at::Tensor>&& zp_fp32,
    c10::optional<at::Tensor>&& bias,
    c10::optional<at::Tensor>&& g_idx,
    c10::optional<int64_t> batch_size,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode) {
  auto op_context = torch_ipex::cpu::detail::woq_linear::create(
      weight,
      weight_dtype,
      weight_shape,
      scales_fp32,
      zp_fp32,
      bias,
      g_idx,
      batch_size,
      group_size,
      lowp_mode,
      act_quant_mode);
  return c10::make_intrusive<IpexWoqLinearOpContext>(
      batch_size, std::move(op_context));
}

at::Tensor IpexWoqLinearOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

at::Tensor IpexWoqLinearOpContext::run(const at::Tensor& input) {
  return torch_ipex::cpu::detail::woq_linear::run(op_context_, input);
}

at::Tensor IpexWoqLinearOpContext::run_eltwise(
    const at::Tensor& input,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm) {
  return torch_ipex::cpu::detail::woq_linear::run_eltwise(
      op_context_, input, post_op, scalars, algorithm);
}

at::Tensor IpexWoqLinearOpContext::run_add(
    const at::Tensor& input,
    const std::vector<at::Tensor>& others) {
  return torch_ipex::cpu::detail::woq_linear::run_add(
      op_context_, input, others);
}

at::Tensor IpexWoqLinearOpContext::run_add_add(
    const at::Tensor& input,
    const std::vector<at::Tensor>& others) {
  return torch_ipex::cpu::detail::woq_linear::run_add_add(
      op_context_, input, others);
}

at::Tensor IpexWoqLinearOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::woq_linear::unpack(op_context_, tensor);
}

at::Tensor IpexWoqLinearOpContext::get_at_packed_weight() {
  return op_context_.at_weight_;
}

c10::optional<at::Tensor> IpexWoqLinearOpContext::get_at_bias() {
  return op_context_.at_bias_;
}

c10::optional<at::Tensor> IpexWoqLinearOpContext::get_g_idx() {
  return op_context_.g_idx_;
}

at::Tensor IpexWoqLinearOpContext::get_scales() {
  if (op_context_.group_size_ > 0 && op_context_.at_weight_.dim() == 4) {
    // [#block_n, #block_k, n_block_size] -> [#block_n, n_block_size, #block_k]
    // -> [N, #block_k]
    auto scales = op_context_.scales_list_[0].permute({0, 2, 1}).contiguous();
    scales = scales.view({-1, scales.size(-1)});
    if (scales.size(0) > op_context_.weight_shape_[0]) {
      return scales.narrow(0, 0, op_context_.weight_shape_[0]);
    }
    return scales;
  }
  if (op_context_.scales_list_[0].size(0) > op_context_.weight_shape_[0]) {
    return op_context_.scales_list_[0].narrow(
        0, 0, op_context_.weight_shape_[0]);
  }
  return op_context_.scales_list_[0];
}

c10::optional<at::Tensor> IpexWoqLinearOpContext::get_zero_points() {
  if (!op_context_.zero_points_list_[0].defined()) {
    return c10::nullopt;
  }
  if (op_context_.group_size_ > 0 && op_context_.at_weight_.dim() == 4) {
    // [#block_n, #block_k, n_block_size] -> [#block_n, n_block_size, #block_k]
    // -> [N, #block_k]
    auto zp = op_context_.zero_points_list_[0].permute({0, 2, 1}).contiguous();
    zp = zp.view({-1, zp.size(-1)});
    if (zp.size(0) > op_context_.weight_shape_[0]) {
      return zp.narrow(0, 0, op_context_.weight_shape_[0]);
    }
    return c10::make_optional(zp);
  }
  if (op_context_.zero_points_list_[0].size(0) > op_context_.weight_shape_[0]) {
    return op_context_.zero_points_list_[0].narrow(
        0, 0, op_context_.weight_shape_[0]);
  }
  return c10::make_optional(op_context_.zero_points_list_[0]);
}

std::vector<int64_t> IpexWoqLinearOpContext::get_weight_shape() {
  return op_context_.weight_shape_;
}

detail::ContextLinearWoq& IpexWoqLinearOpContext::get_context() {
  return op_context_;
}

at::Tensor IpexWoqLinearOpContext::pack(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::woq_linear::pack(op_context_, tensor);
}

void IpexWoqLinearOpContext::load_from_ctx(
    c10::intrusive_ptr<WoqLinearOpContext> other) {
  load_from_ctx_template(this, other);
}
#endif
} // namespace cpu
} // namespace torch_ipex
