#pragma once

#include <cstdint>

namespace gpu {
namespace xetla {

struct MoEGEMMPolicy {
  static constexpr int wg_tile_m = 256;
  static constexpr int wg_tile_n = 256;
  static constexpr int sg_tile_m = 32;
  static constexpr int sg_tile_n = 64;
  static constexpr int k_stride = 16;
  static constexpr int stages = 3;
  static constexpr int sync_freq = 0;
};
} // namespace xetla
} // namespace gpu
