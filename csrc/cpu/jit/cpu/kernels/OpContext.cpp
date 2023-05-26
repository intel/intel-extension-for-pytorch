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

// For weight-only quantization
c10::intrusive_ptr<WoqLinearOpContext> IpexWoqLinearOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size,
    int64_t lowp_mode,
    int64_t num_concats) {
  auto N = weight.size(0);
  const auto qtype = weight.qscheme();
  if (weight.scalar_type() == c10::ScalarType::QInt8) {
    // extract scales from weight
    std::vector<float> weight_scales_float(1, 0.0);
    if (qtype == c10::kPerTensorAffine) {
      weight_scales_float[0] = weight.q_scale();
    } else if (qtype == c10::kPerChannelAffine) {
      weight_scales_float.resize(N, 0.0);
      for (const auto i : c10::irange(N)) {
        weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
      }
    }

    at::Tensor scales = at::empty(
        {static_cast<long>(weight_scales_float.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        weight_scales_float.begin(),
        weight_scales_float.end(),
        scales.data_ptr<float>());

    // extract zero_points from weight
    std::vector<int32_t> weight_zero_points_int32(1, 0);
    if (qtype == c10::kPerTensorAffine) {
      weight_zero_points_int32[0] = weight.q_zero_point();
    } else if (qtype == c10::kPerChannelAffine) {
      weight_zero_points_int32.resize(N, 0);
      for (const auto i : c10::irange(N)) {
        weight_zero_points_int32[i] =
            weight.q_per_channel_zero_points()[i].item<int32_t>();
      }
    }
    at::Tensor zero_points_int32 = at::empty(
        {static_cast<long>(weight_zero_points_int32.size())},
        at::device(c10::kCPU).dtype(c10::kInt));
    std::copy(
        weight_zero_points_int32.begin(),
        weight_zero_points_int32.end(),
        zero_points_int32.data_ptr<int32_t>());

    // convert zero_points from int32_t to float
    std::vector<float> weight_zero_points_float(1, 0);
    if (qtype == c10::kPerTensorAffine) {
      weight_zero_points_float[0] = (float)weight.q_zero_point();
    } else if (qtype == c10::kPerChannelAffine) {
      weight_zero_points_float.resize(N, 0);
      for (const auto i : c10::irange(N)) {
        weight_zero_points_float[i] =
            (float)weight.q_per_channel_zero_points()[i].item<int32_t>();
      }
    }
    at::Tensor zero_points_float = at::empty(
        {static_cast<long>(weight_zero_points_float.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        weight_zero_points_float.begin(),
        weight_zero_points_float.end(),
        zero_points_float.data_ptr<float>());

    auto op_context = torch_ipex::cpu::detail::woq_linear::create(
        weight, scales, zero_points_int32, bias, batch_size, lowp_mode);
    return c10::make_intrusive<IpexWoqLinearOpContext>(
        batch_size,
        std::move(op_context),
        std::move(scales),
        std::move(zero_points_float),
        lowp_mode,
        num_concats);
  } else {
    // extract scales from weight
    std::vector<float> weight_scales_float(1, 0.0);
    if (qtype == c10::kPerChannelAffineFloatQParams) {
      weight_scales_float.resize(N, 0.0);
      for (const auto i : c10::irange(N)) {
        weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
      }
    }

    at::Tensor scales = at::empty(
        {static_cast<long>(weight_scales_float.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        weight_scales_float.begin(),
        weight_scales_float.end(),
        scales.data_ptr<float>());

    // extract zero_points from weight
    std::vector<float> weight_zero_points_float(1, 0);
    if (qtype == c10::kPerChannelAffineFloatQParams) {
      weight_zero_points_float.resize(N, 0);
      for (const auto i : c10::irange(N)) {
        weight_zero_points_float[i] =
            weight.q_per_channel_zero_points()[i].item<float>();
      }
    }
    at::Tensor zero_points_float = at::empty(
        {static_cast<long>(weight_zero_points_float.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        weight_zero_points_float.begin(),
        weight_zero_points_float.end(),
        zero_points_float.data_ptr<float>());
    auto op_context = torch_ipex::cpu::detail::woq_linear::create(
        weight, scales, zero_points_float, bias, batch_size, lowp_mode);
    return c10::make_intrusive<IpexWoqLinearOpContext>(
        batch_size,
        std::move(op_context),
        std::move(scales),
        std::move(zero_points_float),
        lowp_mode,
        num_concats);
  }
}

at::Tensor IpexWoqLinearOpContext::get_data_handle() {
  at::Tensor ptr = at::empty(1, at::kLong);
  ptr[0] = reinterpret_cast<int64_t>(this);
  return ptr;
}

at::Tensor IpexWoqLinearOpContext::run(const at::Tensor& input) {
  return torch_ipex::cpu::detail::woq_linear::run(
      op_context_, scales_list_, zero_points_list_, input, lowp_mode_, num_concats_);
}

at::Tensor IpexWoqLinearOpContext::run_eltwise(
    const at::Tensor& input,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm) {
  return torch_ipex::cpu::detail::woq_linear::run_eltwise(
      op_context_, scales_list_[0], zero_points_list_[0], input,
      post_op, scalars, algorithm, lowp_mode_);
}

at::Tensor IpexWoqLinearOpContext::run_add(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha) {
  return torch_ipex::cpu::detail::woq_linear::run_add(
    op_context_, scales_list_, zero_points_list_, input,
    accumu, alpha, lowp_mode_, num_concats_);
}

at::Tensor IpexWoqLinearOpContext::run_add_relu(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha) {
  return torch_ipex::cpu::detail::woq_linear::run_add_relu(
    op_context_, scales_list_, zero_points_list_, input,
    accumu, alpha, lowp_mode_, num_concats_);
}

at::Tensor IpexWoqLinearOpContext::to_public(const at::Tensor& tensor) {
  return torch_ipex::cpu::detail::woq_linear::unpack(op_context_, tensor);
}

at::Tensor IpexWoqLinearOpContext::get_at_packed_weight() {
  return op_context_.at_weight_;
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

} // namespace cpu
} // namespace torch_ipex
