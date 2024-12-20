#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <libxsmm.h>
#include <torch/all.h>
#include "woq_defines.h"

namespace torch_ipex {
namespace cpu {
using TensorList = std::vector<at::Tensor>;

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
    } else if (qw_type == WOQ_DTYPE_INT4 && sym_quant) {
      // shift from [0, 15] to [-8, 7]
      dqw = (w_int8.to(at::kFloat) - 8) * scale;
    } else {
      dqw = sym_quant ? w_int8.to(at::kFloat) * scale
                      : (w_int8.to(at::kFloat) - zp) * scale;
    }
  } else {
    int64_t num_blocks = scale.size(-2);
    auto rem = K % group_size;
    auto w_fp = qw_type == WOQ_DTYPE_NF4 ? map_nf4_tensor_to_float(w_int8)
                                         : w_int8.to(at::kFloat);
    if (qw_type == WOQ_DTYPE_INT4 && sym_quant) {
      // shift from [0, 15] to [-8, 7]
      w_fp -= 8;
    }
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

// Define this macro to make code more concise
#define CALL_WOQ_KERNEL_IMPL_INT8(T, quant_a_mode) \
  qlinear_woq_affine_impl<                         \
      T,                                           \
      uint8_t,                                     \
      /*TGemmOut*/ float,                          \
      act_type,                                    \
      float,                                       \
      int8_t,                                      \
      quant_a_mode,                                \
      quant_w_mode_>(                              \
      x_quantized,                                 \
      qw,                                          \
      scales_list[fp32_idx],                       \
      biases[fp32_idx],                            \
      y,                                           \
      qw_type,                                     \
      k_splits,                                    \
      fusion_type,                                 \
      others_list,                                 \
      quant_block_k,                               \
      zp_list[int8_idx],                           \
      scale_a_ptr,                                 \
      zp_a_ptr,                                    \
      compensation);

} // namespace cpu
} // namespace torch_ipex