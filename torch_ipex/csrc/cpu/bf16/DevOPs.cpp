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

at::Tensor index(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices) {
  auto&& _tensor = bf16::gen_consistent_tensor(self);
  auto&& _ipex_indices = torch_ipex::bridge::shallowFallbackToCPUTensorList(indices);
  auto&& _ipex_result = at::index(_tensor, _ipex_indices);
  return bf16::gen_mix_prec_tensor(_ipex_result);
}

at::Tensor div(const at::Tensor &self, const at::Tensor &other) {
  auto &&_ipex_self = bf16::gen_consistent_tensor(self);
  auto &&_ipex_other = bf16::gen_consistent_tensor(other);
  auto &&_ipex_result = at::div(_ipex_self, _ipex_other);
  return bf16::gen_mix_prec_tensor(_ipex_result);
}

at::Tensor &div_out(at::Tensor &out, const at::Tensor &self,
                    const at::Tensor &other) {
  auto &&_ipex_out = bf16::gen_consistent_tensor(out);
  auto &&_ipex_self = bf16::gen_consistent_tensor(self);
  auto &&_ipex_other = bf16::gen_consistent_tensor(other);
  at::div_out(_ipex_out, _ipex_self, _ipex_other);
  return out;
}

}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
