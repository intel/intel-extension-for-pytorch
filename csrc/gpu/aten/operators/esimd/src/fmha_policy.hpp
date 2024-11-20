#pragma once

#include <cstdint>
#include "fmha_utils.hpp"

namespace esimd {
// Note: for causal=true mode, query length is supposed to > 1, and mask must be
// true, User is expected to all pass caual mask to mask, along with sliding
// window mask
// For next token phase, the sequence length of query is 1
template <bool kHeadFirst_ = true>
struct PhiSingleQueryPolicy {
  static constexpr uint32_t kHeadSize = 96;
  static constexpr uint32_t kNumHeadsQO = 32;
  static constexpr uint32_t kNumHeadsKV = 32;
  static constexpr uint32_t kSgSeqQO = 1;
  static constexpr uint32_t kSgSeqKV = 8;
  static constexpr uint32_t kNumSgX = 8;
  static constexpr uint32_t kNumSgY = 1;
  static constexpr uint32_t kSimd = 16;
  static constexpr bool kHeadFirst = kHeadFirst_;
  static constexpr bool kUseMask = true;
};

// For first token phase, the sequence length of query is expected to be much
// larger than 1, eg 32in/32out, 1024in/1024out
template <bool kHeadFirst_ = true>
struct PhiMultiQueryPolicy {
  static constexpr uint32_t kHeadSize = 96;
  static constexpr uint32_t kNumHeadsQO = 32;
  static constexpr uint32_t kNumHeadsKV = 32;
  static constexpr uint32_t kSgSeqQO = 4;
  static constexpr uint32_t kSgSeqKV = 16;
  static constexpr uint32_t kNumSgX = 4;
  static constexpr uint32_t kNumSgY = 4;
  static constexpr uint32_t kSimd = 16;
  static constexpr bool kHeadFirst = kHeadFirst_;
  static constexpr bool kUseMask = true;
};

template <bool kHeadFirst_ = true>
struct Phi3SmallSingleQueryPolicy {
  static constexpr uint32_t kHeadSize = 128;
  static constexpr uint32_t kNumHeadsQO = 32;
  static constexpr uint32_t kNumHeadsKV = 8;
  static constexpr uint32_t kSgSeqQO = 1;
  static constexpr uint32_t kSgSeqKV = 8;
  static constexpr uint32_t kNumSgX = 8;
  static constexpr uint32_t kNumSgY = 1;
  static constexpr uint32_t kSimd = 16;
  static constexpr bool kHeadFirst = kHeadFirst_;
  static constexpr bool kUseMask = true;
};

// For first token phase, the sequence length of query is expected to be much
// larger than 1, eg 32in/32out, 1024in/1024out
template <bool kHeadFirst_ = true>
struct Phi3SmallMultiQueryPolicy {
  static constexpr uint32_t kHeadSize = 128;
  static constexpr uint32_t kNumHeadsQO = 32;
  static constexpr uint32_t kNumHeadsKV = 8;
  static constexpr uint32_t kSgSeqQO = 4;
  static constexpr uint32_t kSgSeqKV = 16;
  static constexpr uint32_t kNumSgX = 4;
  static constexpr uint32_t kNumSgY = 4;
  static constexpr uint32_t kSimd = 16;
  static constexpr bool kHeadFirst = kHeadFirst_;
  static constexpr bool kUseMask = true;
};
} // namespace esimd