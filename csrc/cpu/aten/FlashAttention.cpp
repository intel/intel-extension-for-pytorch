#include <torch/all.h>
#include "FlashAttention.h"
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(flash_attention_kernel_stub);

/*
*Caculate the flash attention SDPA. 
*@param query
*@param key
*@param value
*@param scale_attn
*@param attention_mask
*@return attn_outs
*/
at::Tensor flash_attention_forward_cpu(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    const double scale_attn,
    at::Tensor attention_mask){
  return flash_attention_kernel_stub(
      kCPU, query, key, value, scale_attn, attention_mask);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "flash_attention(Tensor query, Tensor key, Tensor value, \
       float scale_attn, Tensor attention_mask)-> Tensor");
  m.impl(
      "flash_attention",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::flash_attention_forward_cpu);
}
}
