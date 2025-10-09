#include "gemm_mxfp4_marlin_impl.h"
#include "../../GEMM_MXFP4_marlin.h"

namespace torch_ipex::xpu::xetla {
template <
    typename dtype_a,
    typename dtype_b,
    typename dtype_c,
    typename dtype_scale,
    typename policy>
cgfs_t XETLA_KERNEL_API hgemm_mxfp4_marlin(
    dtype_c* out,
    const dtype_a* a,
    const dtype_b* b,
    const dtype_scale* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  using hgemm_mxfp4_functor = hgemm_mxfp4_marlin_func<
      dtype_a,
      dtype_b,
      dtype_c,
      dtype_scale,
      policy::periodic_sync_interval,
      policy::prefetch_distance,
      policy::wg_m,
      policy::wg_n,
      policy::sg_m,
      policy::sg_n,
      policy::sg_k,
      policy::dequant_s,
      policy::dequant_mode>;
  return {hgemm_mxfp4_functor::run(
      const_cast<sycl::ext::oneapi::bfloat16*>(a),
      const_cast<uint32_t*>(b),
      out,
      m,
      n,
      k,
      const_cast<uint8_t*>(b_scale))};
}

template cgfs_t XETLA_KERNEL_API
hgemm_mxfp4_marlin<bf16, uint32_t, bf16, uint8_t, GEMV>(
    bf16* out,
    const bf16* a,
    const uint32_t* b,
    const uint8_t* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template cgfs_t XETLA_KERNEL_API
hgemm_mxfp4_marlin<bf16, uint32_t, bf16, uint8_t, GEMV_16>(
    bf16* out,
    const bf16* a,
    const uint32_t* b,
    const uint8_t* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template cgfs_t XETLA_KERNEL_API
hgemm_mxfp4_marlin<bf16, uint32_t, bf16, uint8_t, GEMV_32>(
    bf16* out,
    const bf16* a,
    const uint32_t* b,
    const uint8_t* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template cgfs_t XETLA_KERNEL_API
hgemm_mxfp4_marlin<bf16, uint32_t, bf16, uint8_t, GEMM>(
    bf16* out,
    const bf16* a,
    const uint32_t* b,
    const uint8_t* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);
} // namespace torch_ipex::xpu::xetla
