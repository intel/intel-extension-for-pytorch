#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {
namespace bf16 {

at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);

}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
