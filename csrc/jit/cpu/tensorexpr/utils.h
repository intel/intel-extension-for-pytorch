#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

#include <vector>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

namespace pytnnc = torch::jit::tensorexpr;

c10::MemoryFormat deduce_memory_format(
    c10::IntArrayRef strides,
    c10::IntArrayRef dims);

c10::MemoryFormat deduce_memory_format(
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dims);

std::vector<at::Tensor> constructTensors(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes);

pytnnc::ExprHandle constant(const pytnnc::ArgValue& v);

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
