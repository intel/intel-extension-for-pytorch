#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace cpu {
namespace bf16 {

at::Tensor gen_consistent_tensor(const at::Tensor & self);
at::Tensor gen_mix_prec_tensor(const at::Tensor & self);

}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
