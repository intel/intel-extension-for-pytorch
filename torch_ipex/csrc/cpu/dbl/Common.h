#pragma once

#include <ATen/Tensor.h>

#include "cpu/dil/dil.hpp"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace comm {

dil::tensor dil_tensor_from_dense(const at::Tensor& tensor);
at::Tensor dil_tensor_to_dense(const at::Tensor& tensor);
dil::tensor try_gen_dil_tensor(const at::Tensor &input);
at::Tensor gen_aten_tensor_by(dil::tensor tensor);
at::Tensor empty_dil_tensor(at::IntArrayRef sizes, const at::TensorOptions& options);
void sync_shape_from_dil_to_aten(const at::Tensor& ipex_tensor, const dil::tensor &dil_tensor);

}  // namespace comm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
