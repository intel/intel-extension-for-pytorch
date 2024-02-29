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
    int64_t weight_dtype,
    std::vector<int64_t>&& weight_shape,
    at::Tensor&& scales,
    c10::optional<at::Tensor>&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<at::Tensor>&& g_idx,
    c10::optional<int64_t> batch_size,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode);

c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContextInt4(
    at::Tensor&& weight,
    at::Tensor&& scales,
    at::Tensor&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<at::Tensor>&& g_idx,
    c10::optional<int64_t> batch_size,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode);

at::Tensor woq_linear_run(
    const at::Tensor& input,
    c10::intrusive_ptr<WoqLinearOpContext> op_context);

ContextLinearWoq create(
    at::Tensor& weight,
    int64_t weight_dtype,
    std::vector<int64_t>& weight_shape,
    at::Tensor& scales,
    c10::optional<at::Tensor>& zero_points,
    c10::optional<at::Tensor>& bias,
    c10::optional<at::Tensor>& g_idx,
    const c10::optional<int64_t> batch_size,
    int64_t group_size,
    int64_t lowp_mode,
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

at::Tensor pack(ContextLinearWoq& context, const at::Tensor& tensor);

at::Tensor unpack(ContextLinearWoq& context, const at::Tensor& tensor);

template <bool is_4bit = false>
at::Tensor woq_shuffle_tensor_by_group_idx(
    const at::Tensor& input,
    const std::vector<int64_t>& tensor_shape,
    const at::Tensor& g_idx,
    int64_t group_size);

at::Tensor woq_shuffle_weight_back_by_group_idx(
    const at::Tensor& qweight,
    const std::vector<int64_t>& weight_shape,
    const at::Tensor& g_idx,
    int64_t group_size);

} // namespace woq_linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
#endif