#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "ideep/ideep.hpp"

namespace torch {
namespace jit {

// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace ipex {
static auto matmul_div = Symbol::fromQualString("ipex::matmul_div");
static auto bmm_add = Symbol::fromQualString("ipex::bmm_add");

} // namespace ipex

} // namespace jit
} // namespace torch

namespace torch_ipex {
namespace cpu {

void mkl_fp32_bmm_impl(
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    at::Tensor& out,
    const double& output_scale);

at::Tensor bmm_impl(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor out,
    const ideep::attr_t& attr,
    const std::vector<ideep::tensor>& postop_tensors,
    const float dst_coeff);

at::Tensor dil_matmul(const at::Tensor& left, const at::Tensor& right);

at::Tensor dil_matmul_div(
    const at::Tensor& left,
    const at::Tensor& right,
    at::Tensor out_opt,
    const at::Tensor& div_input);

at::Tensor dil_matmul_div(
    const at::Tensor& left,
    const at::Tensor& right,
    at::Tensor out_opt,
    const c10::Scalar& div_input);

at::Tensor dil_bmm_add(
    const at::Tensor& input,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const c10::Scalar& alpha);

} // namespace cpu
} // namespace torch_ipex
