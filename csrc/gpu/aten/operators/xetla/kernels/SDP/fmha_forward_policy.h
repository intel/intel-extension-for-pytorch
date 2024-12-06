#pragma once

#include "../xetla.h"

namespace gpu::xetla {

struct fmha_policy_base {
  static constexpr uint32_t accum_step = 32;
};

/*
Note:
  kHm / kSgHm == kBc / kSgBc
  kSgHm and kSgBc should be a multiple of 16
  kSgBr should be a multiple of 8
*/

struct fmha_policy_8x128x64 : fmha_policy_base {
  static constexpr uint32_t kBr = 8;
  static constexpr uint32_t kSgBr = 8;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 16;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x128x64 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 64;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_8x256x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 8;
  static constexpr uint32_t kSgBr = 8;
  static constexpr uint32_t kBc = 256;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 16;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_8x512x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 8;
  static constexpr uint32_t kSgBr = 8;
  static constexpr uint32_t kBc = 512;
  static constexpr uint32_t kSgBc = 64;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 16;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x128x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_8x256x256 : fmha_policy_base {
  static constexpr uint32_t kBr = 8;
  static constexpr uint32_t kSgBr = 8;
  static constexpr uint32_t kBc = 256;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x256x256 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 256;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x128x256 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x128x512 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 512;
  static constexpr uint32_t kSgHm = 128;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x64x64 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x64x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 64;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x64x256 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 128;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x64x512 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 512;
  static constexpr uint32_t kSgHm = 128;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_128x128x64 : fmha_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 16;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_128x128x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_128x128x256 : fmha_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 64;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

/* MTL policies */

struct fmha_policy_1x256x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 1;
  static constexpr uint32_t kSgBr = 1;
  static constexpr uint32_t kBc = 256;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 8;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_1x512x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 1;
  static constexpr uint32_t kSgBr = 1;
  static constexpr uint32_t kBc = 512;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 8;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_32x128x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 32;
  static constexpr uint32_t kSgBr = 8;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 16;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 16;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

template <typename fmha_policy, uint32_t block_size>
struct chunked_prefill_policy_wrapper {
  static constexpr uint32_t kBr = fmha_policy::kBr;
  static constexpr uint32_t kSgBr = fmha_policy::kSgBr;
  static constexpr uint32_t kHm = fmha_policy::kHm;
  static constexpr uint32_t subgroup_size =
      kHm / fmha_policy::kSgHm > 4 ? 4 : kHm / fmha_policy::kSgHm;
  static constexpr uint32_t kSgHm = kHm / subgroup_size;
  static constexpr uint32_t thread_num = fmha_policy::thread_num;
  static constexpr uint32_t accum_step = fmha_policy::accum_step;
  static constexpr uint32_t kBc = block_size;
  static constexpr uint32_t kSgBc = block_size / subgroup_size;
};
} // namespace gpu::xetla
