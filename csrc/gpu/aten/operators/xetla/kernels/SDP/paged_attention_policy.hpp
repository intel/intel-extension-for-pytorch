#pragma once

#include <cstdint>

struct paged_attention_base_policy {
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t head_size_stride = 32;
};

template <
    uint32_t max_head_size_,
    uint32_t block_size_,
    uint32_t wg_size_,
    uint32_t max_blocks_per_sg_>
struct paged_attention_policy_v1 : paged_attention_base_policy {
  static constexpr uint32_t wg_size = wg_size_;
  static constexpr uint32_t max_head_size = max_head_size_;
  static constexpr uint32_t block_size = block_size_;
  static constexpr uint32_t partition_size = 0; // 0 for v1
  static constexpr uint32_t max_blocks_per_sg = max_blocks_per_sg_;
};

template <uint32_t max_head_size_, uint32_t block_size_>
struct paged_attention_policy_v2 : paged_attention_base_policy {
  // for attention kernel
  static constexpr uint32_t max_head_size = max_head_size_;
  static constexpr uint32_t block_size = block_size_;
  static constexpr uint32_t partition_size = 512;
  static constexpr uint32_t wg_size =
      partition_size / block_size > 32 ? 32 : partition_size / block_size;
  static constexpr uint32_t max_blocks_per_sg =
      partition_size / (block_size * wg_size);
  // for reduction kernel
  static constexpr uint32_t partition_stride = 8;
  static constexpr uint32_t max_partitions_per_sg = 4;
  static_assert(
      partition_size % (block_size * wg_size) == 0,
      "partition_size should be a multiple of block_size * wg_size");
};