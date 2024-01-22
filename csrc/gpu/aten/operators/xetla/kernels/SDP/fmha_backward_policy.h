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

struct fmha_policy_bwd_64x64x64 : fmha_bwd_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kHm = 64;
  // gemm brbc
  static constexpr uint32_t kBrBc_SgBr = 8;
  static constexpr uint32_t kBrBc_SgBc = 16;
  // gemm brhm
  static constexpr uint32_t kBrHm_SgBr = 8;
  static constexpr uint32_t kBrHm_SgHm = 16;
  // gemm bchm
  static constexpr uint32_t kBcHm_SgBc = 8;
  static constexpr uint32_t kBcHm_SgHm = 16;
  static constexpr uint32_t thread_num =
      (kBr / kBrBc_SgBr) * (kBc / kBrBc_SgBc);
};

struct fmha_policy_bwd_128x128x128 : fmha_bwd_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kHm = 128;
  // gemm brbc
  static constexpr uint32_t kBrBc_SgBr = 16;
  static constexpr uint32_t kBrBc_SgBc = 32;
  // gemm brhm
  static constexpr uint32_t kBrHm_SgBr = 16;
  static constexpr uint32_t kBrHm_SgHm = 32;
  // gemm bchm
  static constexpr uint32_t kBcHm_SgBc = 16;
  static constexpr uint32_t kBcHm_SgHm = 32;
  static constexpr uint32_t thread_num =
      (kBr / kBrBc_SgBr) * (kBc / kBrBc_SgBc);
};
struct fmha_policy_bwd_128x128x256 : fmha_bwd_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kHm = 256;
  // gemm brbc
  static constexpr uint32_t kBrBc_SgBr = 16;
  static constexpr uint32_t kBrBc_SgBc = 32;
  // gemm brhm
  static constexpr uint32_t kBrHm_SgBr = 16;
  static constexpr uint32_t kBrHm_SgHm = 64;
  // gemm bchm
  static constexpr uint32_t kBcHm_SgBc = 16;
  static constexpr uint32_t kBcHm_SgHm = 64;
  static constexpr uint32_t thread_num =
      (kBr / kBrBc_SgBr) * (kBc / kBrBc_SgBc);
};

} // namespace gpu::xetla