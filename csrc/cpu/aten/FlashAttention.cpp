#include "FlashAttention.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(flash_attention_kernel_stub);

/*
 *Caculate the flash attention SDPA with attention mask.
 */
std::tuple<at::Tensor, at::Tensor> flash_attention_forward_cpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    const c10::optional<at::Tensor>& attention_mask,
    c10::optional<double> scale) {
  return flash_attention_kernel_stub(
      kCPU, query, key, value, dropout_p, is_causal, attention_mask, scale);
}

/*
 *Substitude the flash attention SDPA in PT.
 *In order to add optimizations which are hard to upstream, like TPP layout
 *conversion.
 */
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_scaled_dot_product_flash_attention_for_cpu"),
      TORCH_FN((&torch_ipex::cpu::flash_attention_forward_cpu)));
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "flash_attention(Tensor query, Tensor key, Tensor value, \
       float dropout_p=0.0, bool is_causal=False, \
       *, Tensor? attention_mask=None, float? scale=None) -> \
       (Tensor, Tensor)");
  m.impl(
      "flash_attention",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::flash_attention_forward_cpu);
}

} // namespace cpu
} // namespace torch_ipex
