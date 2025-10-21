#include "GEMM/gemm_int4_marlin_impl.h"
#include "../../GEMM_INT4_marlin.h"

namespace torch_ipex::xpu::xetla {
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
    const uint32_t k) {
  using hgemm_wint4_functor = hgemm_wint4_marlin_func<
      dtype_a,
      dtype_b,
      dtype_c,
      dtype_zp,
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
  return {hgemm_wint4_functor::run(
      const_cast<sycl::half*>(a),
      const_cast<uint32_t*>(b),
      out,
      m,
      n,
      k,
      const_cast<sycl::half*>(b_scale))};
}

template cgfs_t XETLA_KERNEL_API
hgemm_wint4_marlin<fp16, uint32_t, fp16, uint32_t, fp16, GEMV>(
    fp16* out,
    const fp16* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const fp16* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template cgfs_t XETLA_KERNEL_API
hgemm_wint4_marlin<fp16, uint32_t, fp16, uint32_t, fp16, GEMV_16>(
    fp16* out,
    const fp16* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const fp16* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template cgfs_t XETLA_KERNEL_API
hgemm_wint4_marlin<fp16, uint32_t, fp16, uint32_t, fp16, GEMV_32>(
    fp16* out,
    const fp16* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const fp16* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);

template cgfs_t XETLA_KERNEL_API
hgemm_wint4_marlin<fp16, uint32_t, fp16, uint32_t, fp16, GEMM>(
    fp16* out,
    const fp16* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const fp16* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k);
} // namespace torch_ipex::xpu::xetla
