#pragma once

#include "xetla.hpp"

namespace gpu::xetla {

struct fmha_policy_base {
  static constexpr uint32_t SIMD = 32;
  static constexpr uint32_t accum_step = 32;
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t sync_freq = 0;
};

/*
Note:
  QueryBlock equals wg_tile0_m
  KeyBlock equals wg_tile0_n and should be a multiple of SIMD
  MaxHeadSize equals wg_tile1_n and should be a multiple of SIMD
  wg_tile1_m must equal wg_tile0_m
*/

struct fmha_policy_f512_t512_h64 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 256; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 16;
  static constexpr uint32_t sg_tile0_n = 32;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 64; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 8;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_h96 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 256; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 16;
  static constexpr uint32_t sg_tile0_n = 32;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 96; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 16;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f384_t384_h64 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 192; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 16;
  static constexpr uint32_t sg_tile0_n = 32;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 64; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 8;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f4096_t4096_h64 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 256; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 16;
  static constexpr uint32_t sg_tile0_n = 32;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 64; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 8;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f4096_t77_h64 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 96; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 16;
  static constexpr uint32_t sg_tile0_n = 32;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 64; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 16;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f1024_t1024_h96 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 256; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 16;
  static constexpr uint32_t sg_tile0_n = 32;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 96; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 16;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f1024_t77_h96 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 96; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 16;
  static constexpr uint32_t sg_tile0_n = 32;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 96; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 32;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_h128 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 256; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 32;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 128; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 32;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f256_t256_h160 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 256; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 32;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 160; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 32;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f256_t77_h160 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 256; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 32;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 160; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 32;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f64_t64_h160 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 64; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 8;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 160; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 32;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f64_t77_h160 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 96; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 16;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 160; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 32;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f64_t64_h256 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 64; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 8;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 64; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 256; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 32;
  static constexpr uint32_t sg_tile1_n = 16;
};

// inference beam<=8
// GPTJ Hs=256
struct fmha_policy_f1_t1024_h256 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 8; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 512; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 8;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 8; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 256; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 8;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f1_t64_h256 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 8; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 64; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 8;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 8; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 256; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 8;
  static constexpr uint32_t sg_tile1_n = 16;
};

// LLAMA/OPT Hs=128
struct fmha_policy_f1_t1024_h128 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 8; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 512; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 8;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 8; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 128; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 8;
  static constexpr uint32_t sg_tile1_n = 16;
};

struct fmha_policy_f1_t64_h128 : fmha_policy_base {
  // gemm0
  static constexpr uint32_t wg_tile0_m = 8; // QueryBlock
  static constexpr uint32_t wg_tile0_n = 64; // KeyBlock
  static constexpr uint32_t sg_tile0_m = 8;
  static constexpr uint32_t sg_tile0_n = 16;
  // gemm1
  static constexpr uint32_t wg_tile1_m = 8; // QueryBlock
  static constexpr uint32_t wg_tile1_n = 128; // MaxHeadSize
  static constexpr uint32_t sg_tile1_m = 8;
  static constexpr uint32_t sg_tile1_n = 16;
};

} // namespace gpu::xetla
