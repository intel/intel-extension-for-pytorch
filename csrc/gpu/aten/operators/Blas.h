#pragma once

#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeXPU {
// res = (m1 * m2.transpose()) / oscale
at::Tensor trans_matmul_div_scalar(
    const at::Tensor& tensor2,
    int64_t dim1,
    int64_t dim2,
    const at::Tensor& tensor1,
    Scalar oscale);

// res = (m1 * m2.transpose()) / oscale + accumul
at::Tensor trans_matmul_div_add(
    const at::Tensor& tensor2,
    int64_t dim1,
    int64_t dim2,
    const at::Tensor& tensor1,
    Scalar oscale,
    Tensor& accumul,
    Scalar alpha);

} // namespace AtenIpexTypeXPU
} // namespace at