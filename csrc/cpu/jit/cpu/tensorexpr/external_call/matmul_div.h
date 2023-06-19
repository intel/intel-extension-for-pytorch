#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

torch::jit::tensorexpr::Tensor computeMatmulDiv(
    const std::vector<torch::jit::tensorexpr::ArgValue>& inputs,
    const std::vector<torch::jit::tensorexpr::ExprHandle>& outputShape,
    const std::vector<torch::jit::tensorexpr::ExprHandle>& outputStride,
    const c10::optional<torch::jit::tensorexpr::ScalarType>& outputType,
    at::Device device);
void nncMatmulDiv(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args);

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
