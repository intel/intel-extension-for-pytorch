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
at::Tensor dil_tensor_to_dense(const dil::tensor& dil_tensor);
dil::tensor try_gen_dil_tensor(const at::Tensor &input);
at::Tensor gen_aten_tensor_by(const dil::tensor& tensor);

}  // namespace comm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
