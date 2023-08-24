#pragma once

#include "xetla.hpp"

namespace gpu::xetla {

struct base_policy {
  static constexpr uint32_t accum_step_bmbc = 64;
  static constexpr uint32_t stages_bmbc = 1;
  static constexpr uint32_t sync_freq_bmbc = 0;

  static constexpr uint32_t accum_step_bmhm = 32;
  static constexpr uint32_t stages_bmhm = 3;
  static constexpr uint32_t sync_freq_bmhm = 0;

  static constexpr uint32_t Beams = 4;
  static constexpr uint32_t kBm = 1;

  static constexpr uint32_t max_load_bytes = 512;
  static constexpr uint32_t reduce_size = 16;
};

/* Note:
  kHm / kSgHm == kBc / kSgBc
  kSgHm and kSgBc should be a multiple of 16
*/

struct ifmha_policy_64x64 : base_policy {
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 16;
};

struct ifmha_policy_128x64 : base_policy {
  static constexpr uint32_t kBc = 256;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 16;
};

struct ifmha_policy_256x64 : base_policy {
  static constexpr uint32_t kBc = 512;
  static constexpr uint32_t kSgBc = 64;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 32;
};
} // namespace gpu::xetla
