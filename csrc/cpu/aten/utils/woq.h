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
    std::fill_n(
        reinterpret_cast<char*>(&brconfig),
        sizeof(libxsmm_gemm_batch_reduce_config),
        0);
    brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

    kernel_func_ = libxsmm_dispatch_brgemm_v2(
        brshape, brflags, /*prefetch_flags=*/0, brconfig);
    std::fill_n(
        reinterpret_cast<char*>(&gemm_param_), sizeof(libxsmm_gemm_param), 0);
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

static constexpr std::array<float, 16> NF4_QUANT_TABLE = {
    -1.0 - 1e-2, // 0b0000
    -0.8480964004993439, // 0b0001
    -0.6106329262256622, // 0b0010
    -0.4599952697753906, // 0b0011
    -0.33967943489551544, // 0b0100
    -0.23460740596055984, // 0b0101
    -0.13791173323988914, // 0b0110
    -0.045525018125772476, // 0b0111
    0.03979014977812767, // 0b1000
    0.1202552504837513, // 0b1001
    0.2035212516784668, // 0b1010
    0.2920137718319893, // 0b1011
    0.3893125355243683, // 0b1100
    0.5016634166240692, // 0b1101
    0.6427869200706482, // 0b1110
    0.8614784181118011, // 0b1111
};

static constexpr std::array<float, 16> NF4_DEQUANT_TABLE = {
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
};

static at::Tensor map_float_tensor_to_nf4(const at::Tensor& t) {
  // Map [-1, 1] to nf4. Assume t in [-1, 1]
  // Logic:
  // for i in range(len(NF4_QUANT_TABLE)):
  //     out_uint8[t > NF4_QUANT_TABLE[i]] = i
  using namespace at::indexing;
  auto out_uint8 = at::empty(t.sizes(), t.options().dtype(at::kByte));
  for (size_t i = 0; i < NF4_QUANT_TABLE.size(); ++i) {
    out_uint8.index_put_({t.greater(NF4_QUANT_TABLE[i])}, i);
  }
  return out_uint8;
}

static at::Tensor map_nf4_tensor_to_float(const at::Tensor& t) {
  // Map nf4 to [-1, 1], t is already unpacked as uint8
  // Logic:
  // for i in range(len(NF4_DEQUANT_TABLE)):
  //     out_dq[t == i] = NF4_DEQUANT_TABLE[i]
  using namespace at::indexing;
  auto out_dq = at::empty(t.sizes(), t.options().dtype(at::kFloat));
  for (size_t i = 0; i < NF4_DEQUANT_TABLE.size(); ++i) {
    out_dq.index_put_({t.eq(i)}, NF4_DEQUANT_TABLE[i]);
  }
  return out_dq;
}

#define WOQ_DTYPE_INT8 1
#define WOQ_DTYPE_INT4 2
#define WOQ_DTYPE_NF4 3

static at::Tensor dequantize_woq_weight(
    const at::Tensor& qw,
    const std::vector<int64_t>& weight_shape,
    const at::Tensor& scale,
    const at::Tensor& zp,
    int64_t qw_type, // weight dtype
    int64_t group_size) {
  using namespace at::indexing;
  static at::Tensor empty_tensor;
  bool sym_quant = qw_type == WOQ_DTYPE_NF4 || !zp.defined();
  TORCH_CHECK(qw.dim() == 2, "Weight must 2D but got ", qw.dim(), "D");
  auto N = weight_shape[0];
  auto K = weight_shape[1];
  at::Tensor w_int8;
  if (qw_type == WOQ_DTYPE_NF4 || qw_type == WOQ_DTYPE_INT4) {
    // unpack to uint8
    w_int8 = at::empty({N, qw.size(1) * 2}, qw.options().dtype(at::kByte));
    w_int8.index({Slice(), Slice(None, None, 2)}).copy_(qw.bitwise_and(0xf));
    w_int8.index({Slice(), Slice(1, None, 2)}).copy_(qw.bitwise_right_shift(4));
  } else { // INT8
    w_int8 = qw;
  }
  at::Tensor dqw;
  if (group_size <= 0) {
    if (qw_type == WOQ_DTYPE_NF4) {
      dqw = map_nf4_tensor_to_float(w_int8) * scale;
    } else {
      dqw = sym_quant ? w_int8.to(at::kFloat) * scale
                      : (w_int8.to(at::kFloat) - zp) * scale;
    }
  } else {
    int64_t num_blocks = scale.size(-2);
    auto rem = K % group_size;
    auto w_fp = qw_type == WOQ_DTYPE_NF4 ? map_nf4_tensor_to_float(w_int8)
                                         : w_int8.to(at::kFloat);
    if (rem > 0) {
      dqw = at::empty({N, K}, qw.options().dtype(at::kFloat));
      auto w_com = w_fp.slice(1, 0, K - rem).view({N, -1, group_size});
      auto w_rem = w_fp.slice(1, K - rem, K).view({N, 1, -1});
      auto s_com = scale.slice(-2, 0, num_blocks - 1);
      auto s_rem = scale.slice(-2, num_blocks - 1, num_blocks);
      auto z_com = sym_quant ? empty_tensor : zp.slice(-2, 0, num_blocks - 1);
      auto z_rem =
          sym_quant ? empty_tensor : zp.slice(-2, num_blocks - 1, num_blocks);
      auto dqw_com = sym_quant ? (w_com * s_com).view({N, -1})
                               : ((w_com - z_com) * s_com).view({N, -1});
      auto dqw_rem = sym_quant ? (w_rem * s_rem).view({N, -1})
                               : ((w_rem - z_rem) * s_rem).view({N, -1});
      dqw.index_put_({Slice(), Slice(None, K - rem)}, dqw_com);
      dqw.index_put_({Slice(), Slice(K - rem, K)}, dqw_rem);
    } else {
      auto w_fp_view = w_fp.view({N, num_blocks, -1});
      dqw = sym_quant ? w_fp_view * scale : (w_fp_view - zp) * scale;
    }
    dqw = dqw.view({N, -1});
  }
  if (K != qw.size(1) * 2) {
    TORCH_CHECK(
        K < qw.size(1) * 2, 'WOQ Linear kernel: Unexpected weight shape');
    dqw = dqw.narrow(1, 0, K).contiguous();
  }
  return dqw;
}

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