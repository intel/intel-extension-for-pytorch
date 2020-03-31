#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "cpu/dil/dil.hpp"
#include "cpu/ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace comm {

dil::tensor dil_tensor_from_dense(const at::Tensor& tensor);
at::Tensor dil_tensor_to_dense(const at::Tensor& tensor);
dil::tensor try_gen_dil_tensor(const at::Tensor &input);
at::Tensor gen_aten_tensor_by(dil::tensor tensor);
at::Tensor empty_dil_tensor(at::IntArrayRef sizes, const at::TensorOptions& options);

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
bool meet_dnnl_route_pre_cond(const at::Tensor& tensor);

bool possible_to_route_to_dnnl(const std::vector<at::Tensor> &tensor_vec);

}  // namespace comm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
