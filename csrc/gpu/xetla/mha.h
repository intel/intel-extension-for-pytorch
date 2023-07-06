#pragma once

#include <stddef.h>

#include <sycl/sycl.hpp>

// namespace xpu {
namespace gpu::xetla {

void fmha_forward_op(
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* out,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys);

void fmha_forward_op_causal(
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* out,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys);

void fmha_forward_op_strided(
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* out,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys);

void fmha_forward_op_causal_strided(
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* out,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys);

void fmha_forward_op_attn_mask_alibi_strided(
    sycl::queue& q,
    void* query,
    void* key,
    void* value,
    void* out,
    void* bias,
    void* alibi,
    void* head_mask,
    const double alpha,
    const double beta,
    const double dropout_p,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys);

} // namespace gpu::xetla
//} // namespace xpu
