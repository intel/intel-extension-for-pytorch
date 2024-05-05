#pragma once
#include <stddef.h>
#include <sycl/sycl.hpp>
#include <xetla_common_types.hpp>
#include "xetla_kernel_api.h"

namespace gpu::xetla {
using namespace xpu::xetla;

enum class XetlaType {
  fp16,
  bf16,
};

struct fmha_forward_kernel_args_t {
  void* query;
  void* key;
  void* value;
  void* alibi;
  void* attn_mask;
  void* dropout;
  void* out;
  void* log_sumexp;
  float alpha;
  float beta;
  float dropout_prob;
  uint32_t num_batches;
  uint32_t num_heads;
  uint32_t num_kv_heads;
  uint32_t head_size;
  uint32_t num_queries;
  uint32_t num_keys;
  uint32_t bias_strideB;
  uint32_t bias_strideN;
  uint32_t bias_strideF;
  uint32_t alibi_padded_block_size;
  uint32_t attn_mask_padded_block_size;
  bool is_causal;
  bool seq_last;
  bool is_training;
  bool is_dropout;
  uint64_t seed_t;
  uint64_t offset_t;
};

// * General interface kernel for FSDP
// * causal
// * permutation t, n, h
// * alibi
XETLA_KERNEL_API cgfs_t fmha_forward_kernel(
    gpu_arch arch,
    XetlaType xeType,
    const fmha_forward_kernel_args_t& args);

XETLA_KERNEL_API cgfs_t fmha_forward_index_kernel(
    void* query,
    void* key,
    void* value,
    void* key_cache,
    void* value_cache,
    int32_t* index,
    void* alibi,
    void* attn_mask,
    uint8_t* dropout,
    void* out,
    uint32_t timestep,
    float alpha,
    float beta,
    float dropout_p,
    uint32_t num_batches,
    uint32_t beam_width,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t num_queries,
    uint32_t num_keys_in,
    uint32_t num_keys_out,
    uint32_t alibi_padding,
    uint32_t attn_mask_padding,
    bool is_causal,
    bool is_bias_bc);

XETLA_KERNEL_API cgfs_t fmha_backward_kernel(
    XetlaType xeType,
    void* grad_out,
    void* query,
    void* key,
    void* value,
    void* bias,
    void* dropout,
    void* out,
    void* log_sumexp,
    void* workspace,
    void* grad_q_tmp,
    float alpha,
    float dropout_prob,
    void* grad_query,
    void* grad_key,
    void* grad_value,
    void* grad_bias,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t bias_strideB,
    uint32_t bias_strideN,
    uint32_t bias_strideF,
    uint32_t attn_mask_padding,
    bool is_causal,
    bool is_dropout,
    uint64_t seed_t,
    uint64_t offset_t);

XETLA_KERNEL_API cgfs_t paged_attention_v1(
    sycl::half* out,
    sycl::half* query,
    sycl::half* key_cache,
    sycl::half* value_cache,
    int32_t* head_mapping,
    int32_t* block_tables,
    int32_t* context_lens,
    float sm_scale,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t block_size,
    uint32_t max_blocks_per_seq,
    uint32_t max_context_len);

XETLA_KERNEL_API cgfs_t paged_attention_v2(
    float* max_logits,
    float* exp_sums,
    sycl::half* tmp_out,
    sycl::half* out,
    sycl::half* query,
    sycl::half* key_cache,
    sycl::half* value_cache,
    int32_t* head_mapping,
    int32_t* block_tables,
    int32_t* context_lens,
    float sm_scale,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t block_size,
    uint32_t max_blocks_per_seq,
    uint32_t max_context_len);
} // namespace gpu::xetla
