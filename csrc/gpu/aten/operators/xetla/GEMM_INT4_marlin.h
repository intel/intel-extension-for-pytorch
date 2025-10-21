#pragma once
#include <stddef.h>
#include <sycl/sycl.hpp>
#include <xetla_common_types.hpp>
#include "xetla_kernel_api.h"

using namespace gpu::xetla;

namespace torch_ipex::xpu::xetla {

class base_config {
 public:
  static constexpr size_t dequant_s = 128;
  static constexpr size_t num_buffer = 1;
  static constexpr DequantMode dequant_mode = DequantMode::FastInterleaved;
  static constexpr uint32_t periodic_sync_interval = 8;
  using data_type_a = sycl::half;
  using data_type_b = unsigned int;
  using data_type_c = sycl::half;
};

class GEMV : public base_config {
 public:
  static constexpr uint32_t prefetch_distance = 6;
  static constexpr size_t wg_m = 8;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 32;
};

class GEMV_16 : public base_config {
 public:
  static constexpr uint32_t prefetch_distance = 6;
  static constexpr size_t wg_m = 16;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 16;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 32;
};

class GEMV_32 : public base_config {
 public:
  static constexpr uint32_t prefetch_distance = 6;
  static constexpr size_t wg_m = 32;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 32;
};

class GEMM : public base_config {
 public:
  static constexpr uint32_t prefetch_distance = 3;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr size_t dequant_s = 128;
  static constexpr size_t num_buffer = 1;
};

template <
    typename dtype_a,
    typename dtype_b,
    typename dtype_c,
    typename dtype_zp,
    typename dtype_scale,
    typename policy>
cgfs_t XETLA_KERNEL_API hgemm_wint4_marlin(
    dtype_c* out,
    const dtype_a* a,
    const dtype_b* b,
    const dtype_zp* b_zp,
    const dtype_scale* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template <
    typename dtype_a,
    typename dtype_b,
    typename dtype_c,
    typename dtype_zp,
    typename dtype_scale,
    typename policy>
cgfs_t XETLA_KERNEL_API group_hgemm_wint4_marlin(
    dtype_c* out,
    const dtype_a* a,
    const dtype_b* b,
    const dtype_zp* b_zp,
    const dtype_scale* b_scale,
    const dtype_a* bias,
    const int* atomic_buffer,
    const int* total_rows_for_each_expert,
    const int expert_num,
    const uint32_t n,
    const uint32_t k);
} // namespace torch_ipex::xpu::xetla
