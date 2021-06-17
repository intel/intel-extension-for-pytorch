#pragma once

#include <array>
#include <vector>

namespace xpu {
namespace dpcpp {

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = LARGE_POOL + 1,
};

using StatArray = std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)>;

struct DeviceStats {
  StatArray allocation;
  StatArray segment;
  StatArray active;
  StatArray inactive_split;
  StatArray allocated_bytes;
  StatArray reserved_bytes;
  StatArray active_bytes;
  StatArray inactive_split_bytes;
  int64_t num_alloc_retries = 0;
  int64_t num_ooms = 0;
};

struct BlockInfo {
  int64_t size = 0;
  bool allocated = false;
  bool active = false;
};

struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  bool is_large = false;
  std::vector<BlockInfo> blocks;
};

}} // namespace xpu::dpcpp
