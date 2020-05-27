#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {
namespace bf16 {
namespace chk {

/**
 * Check if the input tensors can be supported by BF16 OP.
 *
 * @param tensor_vec input tensors.
 * @param dst_dtype Reorder to the data type
 */
bool bf16_support_the_tensors(const std::vector<at::Tensor> &tensor_vec, at::ScalarType dst_dtype);

}  // namespace chk
}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
