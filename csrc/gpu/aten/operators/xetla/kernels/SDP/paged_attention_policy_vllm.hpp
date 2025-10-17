#pragma once

#include <cstdint>

struct paged_attention_base_policy_vllm {
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t head_size_stride = 32;
};

template <
    uint32_t max_head_size_,
    uint32_t block_size_,
    uint32_t query_group_size_>
struct paged_attention_policy_vllm : paged_attention_base_policy_vllm {
  // for attention kernel
  static constexpr uint32_t max_head_size = max_head_size_;
  static constexpr uint32_t block_size = block_size_;
  static constexpr uint32_t partition_size = 512;
  static constexpr uint32_t wg_size =
      partition_size / block_size > 32 ? 32 : partition_size / block_size;
  static constexpr uint32_t query_group_size = query_group_size_;
  static constexpr uint32_t max_blocks_per_sg =
      partition_size / (block_size * wg_size);
  // for reduction kernel
  static constexpr uint32_t partition_stride = 8;
  static constexpr uint32_t max_partitions_per_sg = 4;
  static_assert(
      partition_size % (block_size * wg_size) == 0,
      "partition_size should be a multiple of block_size * wg_size");
  static_assert(
      query_group_size <= wg_size,
      "query_group_size should be less than or equal to wg_size");
};

template <uint32_t block_size_, uint32_t query_group_size_>
struct paged_attention_policy_vllm<64, block_size_, query_group_size_>
    : paged_attention_base_policy_vllm {
  // for attention kernel
  static constexpr uint32_t max_head_size = 64;
  static constexpr uint32_t block_size = block_size_;
  static constexpr uint32_t partition_size = 256;
  static constexpr uint32_t wg_size =
      partition_size / block_size > 32 ? 32 : partition_size / block_size;
  static constexpr uint32_t query_group_size = query_group_size_;
  static constexpr uint32_t max_blocks_per_sg =
      partition_size / (block_size * wg_size);
  // for reduction kernel
  static constexpr uint32_t partition_stride = 8;
  static constexpr uint32_t max_partitions_per_sg = 16;
  static_assert(
      partition_size % (block_size * wg_size) == 0,
      "partition_size should be a multiple of block_size * wg_size");
};

template <
    uint32_t max_head_size_,
    uint32_t block_size_,
    uint32_t query_group_size_>
struct paged_attention_loop_policy_vllm : paged_attention_base_policy_vllm {
  static constexpr uint32_t head_size_stride = 16;
  static constexpr uint32_t max_head_size = max_head_size_;
  static constexpr uint32_t block_size = block_size_;
  static constexpr uint32_t partition_size = 512;
  static constexpr uint32_t wg_size =
      partition_size / block_size > 32 ? 32 : partition_size / block_size;
  static constexpr uint32_t query_group_size = query_group_size_;
  static constexpr uint32_t max_blocks_per_sg =
      partition_size / (block_size * wg_size);
  static_assert(
      partition_size % (block_size * wg_size) == 0,
      "partition_size should be a multiple of block_size * wg_size");
  static_assert(
      query_group_size <= wg_size,
      "query_group_size should be less than or equal to wg_size");
};

template <uint32_t block_size_, uint32_t query_group_size_>
struct paged_attention_loop_policy_vllm<64, block_size_, query_group_size_>
    : paged_attention_base_policy_vllm {
  static constexpr uint32_t head_size_stride = 16;
  static constexpr uint32_t max_head_size = 64;
  static constexpr uint32_t block_size = block_size_;
  static constexpr uint32_t partition_size = 256;
  static constexpr uint32_t wg_size =
      partition_size / block_size > 32 ? 32 : partition_size / block_size;
  static constexpr uint32_t query_group_size = query_group_size_;
  static constexpr uint32_t max_blocks_per_sg =
      partition_size / (block_size * wg_size);
  static_assert(
      partition_size % (block_size * wg_size) == 0,
      "partition_size should be a multiple of block_size * wg_size");
};
