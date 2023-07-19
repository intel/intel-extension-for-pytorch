#pragma once

#include "xetla.hpp"

namespace gpu::xetla {
/*
Note:
  kHm / kSgHm == kBc / kSgBc
  kSgHm and kSgBc should be a multiple of 16
*/

struct ifmha_policy_64x64 {
  static constexpr uint32_t accum_step = 32;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 16;
};

struct ifmha_policy_128x64 {
  static constexpr uint32_t accum_step = 32;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 32;
};

struct ifmha_policy_256x64 {
  static constexpr uint32_t accum_step = 32;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 32;
};

} // namespace gpu::xetla
