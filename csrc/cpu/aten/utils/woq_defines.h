#pragma once

#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

#define WOQ_DTYPE_INT8 1
#define WOQ_DTYPE_INT4 2
#define WOQ_DTYPE_NF4 3

#define UNQUANT_A -1
#define QUANT_A_PER_TENSOR 0
#define QUANT_A_PER_K_BLOCK 1
#define QUANT_A_PER_M 2
#define QUANT_A_PER_M_K_BLOCK 3
#define QUANT_A_PER_TENSOR_SYM 4
#define QUANT_A_PER_K_BLOCK_SYM 5
#define QUANT_A_PER_M_SYM 6
#define QUANT_A_PER_M_K_BLOCK_SYM 7

#define LOWP_MODE_NONE 0
#define LOWP_MODE_FP16 1
#define LOWP_MODE_BF16 2
#define LOWP_MODE_INT8 3

// Fusion types
#define WOQ_FUSE_NONE 0x0
// Unary post ops
#define WOQ_FUSE_GELU_ERF 0x1
#define WOQ_FUSE_GELU_TANH 0x2
#define WOQ_FUSE_RELU 0x3
#define WOQ_FUSE_SILU 0x4
// Binary post ops
#define WOQ_FUSE_ADD 0x10
#define WOQ_FUSE_ADD_ADD 0x20
#define WOQ_FUSE_MUL 0x30

// weight quant mode
#define QUANT_W_PER_CHANNEL 0
#define QUANT_W_PER_K_BLOCK 1
#define QUANT_W_PER_CHANNEL_SYM 2
#define QUANT_W_PER_K_BLOCK_SYM 3

#define WOQ_N_BLOCK_SIZE 32

constexpr bool is_asymmetric_quant_a(const int quant_a_mode) {
  return quant_a_mode <= QUANT_A_PER_M_K_BLOCK;
}

constexpr bool is_asymmetric_quant_w(const int quant_w_mode) {
  return quant_w_mode <= QUANT_W_PER_K_BLOCK;
}

constexpr bool is_4bit(const int qw_type) {
  return qw_type == WOQ_DTYPE_INT4 || qw_type == WOQ_DTYPE_NF4;
}
