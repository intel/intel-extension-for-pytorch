#pragma once

#include "../xetla.h"

namespace xpu {
namespace xetla {

class gru_base_config {
 public:
  static constexpr uint32_t wg_tile_m = 8;
  static constexpr uint32_t wg_tile_n = 1024;
  static constexpr uint32_t sg_tile_m = 8;
  static constexpr uint32_t sg_tile_n = 32;
  static constexpr uint32_t sg_tile_k = 16;
};

class m512k92n256_fwd : public gru_base_config {
 public:
  using input_T = bf16;
  using Act_T = float;
  static constexpr uint32_t wg_tile_m = 8;
  static constexpr uint32_t wg_tile_n = 512;
  static constexpr uint32_t sg_tile_m = 8;
  static constexpr uint32_t sg_tile_n = 16;
  static constexpr uint32_t sg_tile_k_0 = 16;
  static constexpr uint32_t sg_tile_k_1 = 16;
};

class m512k379n681_fwd : public gru_base_config {
 public:
  using input_T = bf16;
  using Act_T = float;
  static constexpr uint32_t wg_tile_m = 16;
  static constexpr uint32_t wg_tile_n = 1024;
  static constexpr uint32_t sg_tile_m = 16;
  static constexpr uint32_t sg_tile_n = 32;
  static constexpr uint32_t sg_tile_k_0 = 16;
  static constexpr uint32_t sg_tile_k_1 = 16;
};

class m512k379n681_bpk : public gru_base_config {
 public:
  using input_T = bf16;
  using Act_T = float;
  static constexpr uint32_t wg_tile_n_0 = 256;
  static constexpr uint32_t sg_tile_n_0 = 32;
  static constexpr uint32_t wg_tile_n_1 = 128;
  static constexpr uint32_t sg_tile_n_1 = 16;
  static constexpr uint32_t wg_tile_m = 64;
  static constexpr uint32_t sg_tile_m = 16;
  static constexpr uint32_t sg_tile_k = 16;
};

class m512k92n256_bpk : public gru_base_config {
 public:
  using input_T = bf16;
  using Act_T = float;
  static constexpr uint32_t wg_tile_n_0 = 128;
  static constexpr uint32_t sg_tile_n_0 = 32;
  static constexpr uint32_t wg_tile_n_1 = 64;
  static constexpr uint32_t sg_tile_n_1 = 16;
  static constexpr uint32_t wg_tile_m = 32;
  static constexpr uint32_t sg_tile_m = 8;
  static constexpr uint32_t sg_tile_k = 16;
};

class m512k379n681_bpi : public gru_base_config {
 public:
  using input_T = bf16;
  using Act_T = float;
  static constexpr uint32_t wg_tile_m = 8;
  static constexpr uint32_t wg_tile_n_0 = 1024;
  static constexpr uint32_t sg_tile_n_0 = 32;
  static constexpr uint32_t wg_tile_n_1 = 512;
  static constexpr uint32_t sg_tile_n_1 = 16;
  static constexpr uint32_t sg_tile_m = 8;
  static constexpr uint32_t sg_tile_k = 32;
};

class m512k92n256_bpi : public gru_base_config {
 public:
  using input_T = bf16;
  using Act_T = float;
  static constexpr uint32_t wg_tile_m = 8;
  static constexpr uint32_t wg_tile_n_0 = 1024;
  static constexpr uint32_t sg_tile_n_0 = 32;
  static constexpr uint32_t wg_tile_n_1 = 512;
  static constexpr uint32_t sg_tile_n_1 = 16;
  static constexpr uint32_t sg_tile_m = 8;
  static constexpr uint32_t sg_tile_k = 32;
};

} // namespace xetla
} // namespace xpu
