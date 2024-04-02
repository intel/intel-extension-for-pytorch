#pragma once

#ifdef USE_LIBXSMM
#include <dyndisp/DispatchStub.h>
#include <libxsmm.h>
#include <torch/all.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_map>
#include "utils.h"

namespace torch_ipex {
namespace cpu {
namespace {

#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

// Get mask for last column
template <int EXPANDED_N, int col>
constexpr inline unsigned short get_mask(unsigned short mask) {
  // Not last column, return 0xffffff indicating load/store all 16 floats
  if constexpr (col < EXPANDED_N / 16 - 1)
    return (unsigned short)0xffff;
  else
    return mask;
}
template <int EXPANDED_N>
constexpr inline unsigned short get_mask(int col, unsigned short mask) {
  // Not last column, return 0xffffff indicating load/store all 16 floats
  if (col < EXPANDED_N / 16 - 1)
    return (unsigned short)0xffff;
  else
    return mask;
}

const int BLOCK_N = 64, BLOCK_K = 96, PREFETCH_K = 64;

struct DotMicroKernelKey {
  bool trans_a;
  bool trans_b;
  int lda;
  int ldb;
  int ldc;

  DotMicroKernelKey(bool trans_a, bool trans_b, int lda, int ldb, int ldc)
      : trans_a(trans_a), trans_b(trans_b), lda(lda), ldb(ldb), ldc(ldc) {}

  bool operator==(const DotMicroKernelKey& other) const {
    return trans_a == other.trans_a && trans_b == other.trans_b &&
        lda == other.lda && ldb == other.ldb && ldc == other.ldc;
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
class DotMicroKernel {
 public:
  DotMicroKernel(bool trans_a, bool trans_b, int lda, int ldb, int ldc) {
    libxsmm_gemm_shape brshape = libxsmm_create_gemm_shape(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        lda,
        ldb,
        ldc,
        /*type A*/ LIBXSMM_DATATYPE_F32,
        /*type B*/ LIBXSMM_DATATYPE_F32,
        /*type C*/ LIBXSMM_DATATYPE_F32,
        /*acctype*/ LIBXSMM_DATATYPE_F32);
    libxsmm_bitfield brflags =
        (trans_a ? LIBXSMM_GEMM_FLAG_TRANS_A : LIBXSMM_GEMM_FLAG_NONE) |
        (trans_b ? LIBXSMM_GEMM_FLAG_TRANS_B : LIBXSMM_GEMM_FLAG_NONE);
    libxsmm_gemm_batch_reduce_config brconfig;
    memset(&brconfig, 0, sizeof(libxsmm_gemm_batch_reduce_config));
    brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

    kernel_func_ = libxsmm_dispatch_brgemm_v2(
        brshape, brflags, /*prefetch_flags=*/0, brconfig);
    memset(&gemm_param_, 0, sizeof(libxsmm_gemm_param));
  }

  void operator()(void* A, void* B, void* C) {
    gemm_param_.a.primary = (void*)A;
    gemm_param_.b.primary = (void*)B;
    gemm_param_.c.primary = (void*)C;
    kernel_func_(&gemm_param_);
  }

 private:
  libxsmm_gemmfunction kernel_func_;
  libxsmm_gemm_param gemm_param_;
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
using DotMicroKernelRef =
    std::shared_ptr<DotMicroKernel<BLOCK_M, BLOCK_N, BLOCK_K>>;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
DotMicroKernelRef<BLOCK_M, BLOCK_N, BLOCK_K> create_or_get_dot_microkernel(
    bool trans_a,
    bool trans_b,
    int lda,
    int ldb,
    int ldc) {
  thread_local std::unordered_map<
      DotMicroKernelKey,
      DotMicroKernelRef<BLOCK_M, BLOCK_N, BLOCK_K>>
      cache;
  DotMicroKernelKey key(trans_a, trans_b, lda, ldb, ldc);
  auto search = cache.find(key);
  if (search != cache.end()) {
    return search->second;
  } else {
    cache.insert(
        {key,
         std::make_shared<DotMicroKernel<BLOCK_M, BLOCK_N, BLOCK_K>>(
             trans_a, trans_b, lda, ldb, ldc)});
    return cache[key];
  }
}
} // namespace
} // namespace cpu
} // namespace torch_ipex

namespace std {
template <>
struct hash<torch_ipex::cpu::DotMicroKernelKey> {
  std::size_t operator()(const torch_ipex::cpu::DotMicroKernelKey& key) const {
    std::size_t h = std::hash<bool>()(key.trans_a);
    h = std::hash<bool>()(key.trans_b) ^ (h << 1);
    h = std::hash<int>()(key.lda) ^ (h << 1);
    h = std::hash<int>()(key.ldb) ^ (h << 1);
    h = std::hash<int>()(key.ldc) ^ (h << 1);
    return h;
  }
};
} // namespace std

#endif