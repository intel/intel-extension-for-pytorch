#include "torch_ipex/csrc/cpu/bf16/DevOPs.hpp"

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/cpu/dbl/Common.h"
#include "torch_ipex/csrc/cpu/ShadeDataContext.h"
#include "torch_ipex/csrc/cpu/bf16/Bridge.hpp"

namespace torch_ipex {
namespace cpu {
namespace bf16 {

at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index) {
  auto&& _tensor = bf16::gen_consistent_tensor(self);
  auto&& _ipex_index = bf16::gen_consistent_tensor(index);
  auto&& _ipex_result = at::index_select(_tensor, dim, _ipex_index);
  return bf16::gen_mix_prec_tensor(_ipex_result);
}

at::Tensor index(const at::Tensor & self, at::TensorList indices) {
  auto&& _tensor = bf16::gen_consistent_tensor(self);
  auto&& _ipex_indices = bridge::shallowFallbackToCPUTensorList(indices);
  auto&& _ipex_result = at::index(_tensor, _ipex_indices);
  return bf16::gen_mix_prec_tensor(_ipex_result);
}

}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
