#pragma once

#include <ATen/Tensor.h>

#include "cpu/dil/dil.hpp"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace comm {

void reorder_to_bf16_for_mix_prec(const at::Tensor& tensor);

/**
 * Reorder the input tensor to the specified scalar type.
 *
 * @param[in] tensor          The tensor to be reordered to the spcified scalar type
 * @param[in] dst_scalar_type The scalar type which the shade buffer of the ipex tensor will be reordered to. It should
 *                            be at::kBFloat16 or at::kFloat
 */
void reorder_to_dtype(const at::Tensor& tensor, at::ScalarType dtype);

/**
 * Reorder the DNNL tensor to the public format if the input tensor contains DNNL tensor.
 *
 * @param[in] tensor The DNNL tensor of the input ipex tensor to be reordered to public format
 */
void reorder_to_public(const at::Tensor& tensor);

/**
 * Reorder the input tensor to the expected descriptor.
 *
 * @param[in] tensor        The tensor to be reordered to the spcified oneDNN descriptor
 * @param[in] expected_desc The dil buffer of the input tensor will be reordered to expected_desc
 */
void reorder_to_desc(const at::Tensor& tensor, const dil::tensor::desc& expected_desc);

/**
 * Replace the whole original storage with a dil storage `dil_buffer`
 * @param[in] tensor            The input tensor
 * @param[in] dil_tensor_buffer The dil tensor buffer
 */
void equip_dil_buffer(const at::Tensor& tensor, dil::tensor dil_buffer);

dil::tensor try_gen_dil_tensor(const at::Tensor &input);
at::Tensor gen_aten_tensor_by(dil::tensor&& tensor);
at::Tensor empty_dil_tensor(at::IntArrayRef sizes, const at::TensorOptions& options);
void sync_shape_from_dil_to_aten(const at::Tensor& ipex_tensor, const dil::tensor &dil_tensor);
std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param, const char *param_name, int64_t expected_dim);

}  // namespace comm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
