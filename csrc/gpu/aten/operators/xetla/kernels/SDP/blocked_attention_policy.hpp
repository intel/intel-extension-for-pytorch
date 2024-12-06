#pragma once

#include <cstdint>

struct blocked_attention_base_policy {
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t head_size_stride = 32;
};

template <
    uint32_t max_head_size_,
    uint32_t block_size_,
    uint32_t wg_size_,
    uint32_t max_blocks_per_sg_>
struct chunked_prefill_slice_kv_policy : blocked_attention_base_policy {
  static constexpr uint32_t wg_size = wg_size_;
  static constexpr uint32_t max_head_size = max_head_size_;
  static constexpr uint32_t block_size = block_size_;
  static constexpr uint32_t max_blocks_per_sg = max_blocks_per_sg_;
  static constexpr uint32_t block_m = 8;
};

template <uint32_t max_head_size_, uint32_t block_size_>
struct chunked_prefill_split_kv_policy : blocked_attention_base_policy {
  static constexpr uint32_t max_head_size = max_head_size_;
  static constexpr uint32_t block_size = block_size_;
  static constexpr uint32_t block_m = 8;
  static constexpr uint32_t partition_size = 512;
  static constexpr uint32_t wg_size =
      partition_size / block_size > 32 ? 32 : partition_size / block_size;
  static constexpr uint32_t max_blocks_per_sg =
      partition_size / (block_size * wg_size);

  static constexpr uint32_t partition_stride = 8;
  static constexpr uint32_t max_partitions_per_sg = 4;
  static_assert(
      partition_size % (block_size * wg_size) == 0,
      "partition_size should be a multiple of block_size * wg_size");
};