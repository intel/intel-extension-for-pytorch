#include "MultiHeadAttention.h"
#include <torch/all.h>

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(bert_mha_kernel_stub);
IPEX_DEFINE_DISPATCH(sd_mha_kernel_v1_stub);
IPEX_DEFINE_DISPATCH(sd_mha_kernel_v2_stub);

at::Tensor bert_flash_mha(
    const at::Tensor& qkv,
    const at::Tensor& rel_kv,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& dim_per_head) {
  return bert_mha_kernel_stub(
      kCPU, qkv, rel_kv, head_num, headSize, dim_per_head);
}

at::Tensor sd_flash_mha(
    const at::Tensor& qkv,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& scale) {
  return sd_mha_kernel_v1_stub(kCPU, qkv, head_num, headSize, scale);
}

at::Tensor sd_flash_mha(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& scale) {
  return sd_mha_kernel_v2_stub(
      kCPU, query, key, value, head_num, headSize, scale);
}

} // namespace cpu
} // namespace torch_ipex
