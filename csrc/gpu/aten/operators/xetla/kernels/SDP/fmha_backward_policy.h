#pragma once

#include "xetla.hpp"

namespace gpu::xetla {

struct fmha_bwd_policy_base {
  static constexpr uint32_t accum_step = 16;
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t sync_freq = 0;
};

/*
Note:
  kHm / kSgHm == kBc / kSgBc
  kSgHm and kSgBc should be a multiple of 16
  kSgBr should be a multiple of 8
*/

struct fmha_bwd_policy_128x128x64 : fmha_bwd_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kBcHm_SgBc = 16;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 16;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_bwd_policy_128x128x128 : fmha_bwd_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kBcHm_SgBc = 16;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_bwd_policy_128x128x256 : fmha_bwd_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kBcHm_SgBc = 16;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 64;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_bwd_policy_64x128x512 : fmha_bwd_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kBcHm_SgBc = 32;
  static constexpr uint32_t kHm = 512;
  static constexpr uint32_t kSgHm = 128;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

} // namespace gpu::xetla