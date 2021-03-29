#pragma once

#include <ATen/ATen.h>

#include "cpu/dil/dil.hpp"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace comm {

/**
 * Reorder the input tensor to bf16 for mix precision
 * If not_reorder_for_training is set to true,
 * do not reorder the tensor when doing training in mix precision mode
 * @param[in] tensor                   The input tensor
 * @param[in] not_reorder_for_training Set to true if the tensor should not be reordered during training in mix precision mode
 */
void reorder_to_bf16_for_mix_prec(const at::Tensor& tensor, bool not_reorder_for_training = false);

std::vector<std::vector<float>>
get_int8_scales(const at::TensorList &tensor, bool uint8_used, int64_t ops_id);

bool get_int8_quantized_status(const int64_t ops_id);

void reorder_to_int8_for_mix_prec(const at::Tensor& tensor, std::vector<float> scales, bool uint8_used = false);

/**
 * Reorder the input tensor to the specified scalar type.
 *
 * @param[in] tensor          The tensor to be reordered to the spcified scalar type
 * @param[in] dst_scalar_type The scalar type which the shade buffer of the ipex tensor will be reordered to. It should
 *                            be at::kBFloat16 or at::kFloat
 */
void reorder_to_dtype(const at::Tensor& tensor, at::ScalarType dtype, std::vector<float> sclaes = {});

/**
 * Reorder (outplace) the dil input tensor to the specified dil data type.
 *
 * @param[in] dil_tensor The dil tensor to be reordered to the spcified dil data type
 * @param[in] dtype      The dil data type which the dil tensor will be reordered to.
 */
dil::tensor reorder_dil_tensor_to_dtype(const dil::tensor &dil_tensor, dil::data_type dtype);

/**
 * Reorder the DNNL tensor to the public format if the input tensor contains DNNL tensor.
 *
 * @param[in] tensor The DNNL tensor of the input ipex tensor to be reordered to public format
 */
void reorder_to_public(const at::Tensor &tensor, bool remain_dtype = false);

/**
 * Reorder the input tensor to the expected descriptor.
 *
 * @param[in] tensor        The tensor to be reordered to the spcified oneDNN descriptor
 * @param[in] expected_desc The dil buffer of the input tensor will be reordered to expected_desc
 */
void reorder_to_desc(const at::Tensor& tensor, const dil::tensor::desc& expected_desc, const std::vector<float> scales = {});

/**
 * Replace the whole original storage with a dil storage `dil_buffer`
 * @param[in] tensor            The input tensor
 * @param[in] dil_tensor_buffer The dil tensor buffer
 * @param[in] padding_size      The padded size of the dil_buffer ( = storage size calculated using dims and strides - numel())
 */
void equip_dil_buffer(const at::Tensor& tensor, dil::tensor dil_buffer, int64_t padding_size = 0);

dil::tensor try_gen_dil_tensor(const at::Tensor& input);
dil::tensor try_gen_dil_tensor(const at::Tensor &input, const dil::tensor::desc& desc);

/**
 * TODO: only for LSTM, may need to rewrite later
 */
dil::tensor try_gen_dil_tensor(const at::Tensor &input, const std::vector<int64_t> desc_size, const dil::format_tag dil_format_tag);
dil::tensor try_gen_dil_storage(const at::Tensor& input);
at::Tensor gen_aten_tensor_by(dil::tensor&& tensor);
dil::tensor dil_tensor_from_cpu_buffer(const at::Tensor& tensor);
dil::tensor dil_tensor_from_cpu_buffer(const at::Tensor& tensor, dil::deleter_ptr deleter_fn);

/**
 * TODO: only for LSTM, may need to rewrite later
 */
dil::tensor dil_tensor_from_cpu_buffer(const at::Tensor& tensor, const std::vector<int64_t> desc_size, const dil::format_tag dil_format_tag);
at::Tensor empty_dil_tensor(at::IntArrayRef sizes, const at::TensorOptions& options);
void sync_shape_from_dil_to_aten(const at::Tensor& ipex_tensor, const dil::tensor &dil_tensor);
std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param, const char *param_name, int64_t expected_dim);

at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g);
}  // namespace comm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
