#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace chk {

/**
 * Check if the input tensors can be supported by DNNL non-in-place OP.
 *
 * @param tensor_vec input tensors.
 */
bool dnnl_support_the_tensors(const std::vector<at::Tensor> &tensor_vec);

/**
 * Check if the input tensors can be supported by DNNL in-place OP.
 *
 * @param tensor_vec input tensors.
 */
bool dnnl_inplace_support_the_tensors(const std::vector<at::Tensor> &tensor_vec);

/**
 * Check if current tensor can be routed to DNNL OP or not.
 *
 * @param tensor an aten tensor.
 *
 * @return If the input tensor is contiguous with plain format and its device
 * is DPCPP, then the tensor can be routed to DNNL OP. Otherwise, the tensor
 * will be fallbacked to CPU.
 *
 * @note Current condition may be too critical. The DEVICE contidion could be
 * eliminated.
 */
bool dnnl_support_the_memory_layout_of(const at::Tensor& tensor);

/**
 * Check if the input tenosrs can be routed to DNNL OP. The input tensor should
 * meet the conditions defined by @ref dnnl_support_the_memory_layout_of
 *
 * @param tensor_vec input tensors to be checked if it can be routed to DNNL
 *
 */
bool dnnl_support_the_memory_layout_of(const std::vector<at::Tensor> &tensor_vec);

/**
 * Check if the data type of the input tenosrs can be supported by DNNL
 *
 * @param tensor_vec input tensors
 *
 */
bool dnnl_support_the_data_type_of(const std::vector<at::Tensor> &tensor_vec);

/**
 * Check if the dimension of the input tenosrs can be supported by DNNL. The dimension
 * of the input tensor should be > 0.
 *
 * @param tensor_vec input tensors
 *
 */
bool dnnl_support_the_dimension_of(const std::vector<at::Tensor> &tensor_vec);

/**
 * Check if all input tensors has data
 *
 * @param tensor_vec input tensors
 *
 */
static inline bool dnnl_tensor_has_data(const std::vector<at::Tensor> &tensor_vec);

/**
 * Check if all input tensors are dpcpp tensor
 *
 * @param tensor_vec input tensors
 *
 */
bool all_is_dpcpp(const std::vector<at::Tensor> &tensor_vec);

}  // namespace chk
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
