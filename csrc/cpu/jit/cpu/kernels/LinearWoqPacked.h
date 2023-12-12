#pragma once
#ifdef USE_LIBXSMM
#include <ATen/Tensor.h>
#include "ContextLinearWoq.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace woq_linear {

// WOQ = weight-only quantization
c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContext(
    at::Tensor&& weight,
    std::vector<int64_t>&& weight_shape,
    at::Tensor&& scales,
    at::Tensor&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size,
    bool is_int4,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t num_concats,
    int64_t act_quant_mode);

c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContextInt4(
    at::Tensor&& weight,
    at::Tensor&& scales,
    at::Tensor&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t num_concats,
    int64_t act_quant_mode);

at::Tensor woq_linear_run(
    const at::Tensor& input,
    c10::intrusive_ptr<WoqLinearOpContext> op_context);

ContextLinearWoq create(
    at::Tensor& weight,
    std::vector<int64_t>& weight_shape,
    at::Tensor& scales,
    at::Tensor& zero_points,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size,
    bool is_int4,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t num_concats,
    int64_t act_quant_mode);

at::Tensor run(ContextLinearWoq& context, const at::Tensor& input);

at::Tensor run_eltwise(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm);

at::Tensor run_add(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const std::vector<at::Tensor>& others);

at::Tensor run_add_add(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const std::vector<at::Tensor>& others);

at::Tensor woq_linear_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const at::Tensor& op_context);

at::Tensor woq_linear_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const at::Tensor& op_context);

at::Tensor pack(ContextLinearWoq& context, const at::Tensor& tensor);

at::Tensor unpack(ContextLinearWoq& context, const at::Tensor& tensor);

} // namespace woq_linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
#endif