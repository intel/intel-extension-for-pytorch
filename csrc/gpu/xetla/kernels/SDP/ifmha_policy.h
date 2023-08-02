#pragma once

#include "xetla.hpp"

namespace gpu::xetla {
/*
Note:
  1. accum_step is the number of elements loaded for the key (K) and value (V)
matrices for each thread during each iteration.
  2. kHm / kSgHm == kBc / kSgBc
  3. kHm should be a multiple of accum_step
  4. kSgBc must be less than 32, due to limitation for 2d load of index
*/

struct base_policy {
  static constexpr uint32_t prefetch_distance = 3;
};

struct ifmha_policy_64x64 : base_policy {
  static constexpr uint32_t accum_step = 32;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 16;
};

struct ifmha_policy_128x64 : base_policy {
  static constexpr uint32_t accum_step = 32;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 32;
};

// for small batch size
struct ifmha_policy_s_256x64 : base_policy {
  static constexpr uint32_t accum_step = 256;
  static constexpr uint32_t kBc = 512;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 512;
  static constexpr uint32_t kSgHm = 16;
};

// for large batch size
struct ifmha_policy_l_256x64 : base_policy {
  static constexpr uint32_t accum_step = 256;
  static constexpr uint32_t kBc = 512;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 16;
};

} // namespace gpu::xetla
