#pragma once

#include "../../DEQUANTIZE_INT4.h"
#include "../xetla.h"

namespace torch_ipex::xpu::xetla {

template <typename kernel>
struct XetlaInt4DequantizeRunFunctor {
  KERNEL_MAIN void operator()(nd_item<3> item) const {
    kernel::call(item, args_);
  }
  XetlaInt4DequantizeRunFunctor(typename kernel::arguments_t args)
      : args_(args) {}

 private:
  typename kernel::arguments_t args_;
};

// TODO(zhe): support asym weight. may need to handle other compressed weight in
// the future(e.g. int4x2)
template <
    typename scalar_t,
    quant_mode q_mode,
    int WG_N,
    int WG_K,
    int SG_N,
    int SG_K,
    int K_STRIDE,
    int DEQUANT_S,
    int ARCH>
XETLA_KERNEL_API cgf_t xetla_dequantize_int4_weight(
    scalar_t* out,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const uint32_t n,
    const uint32_t k) {
  using int4_dequantize_attr = gpu::xetla::kernel::
      int4_dequantize_attr_t<WG_N, WG_K, SG_N, SG_K, K_STRIDE>;
  static constexpr gpu::xetla::mem_layout mat_b_layout =
      gpu::xetla::mem_layout::col_major;
  static constexpr quant_info q_info{q_mode, DEQUANT_S, mat_b_layout};

  using int4_dequantize_kernel = gpu::xetla::kernel::int4_dequantize_t<
      int4x8,
      scalar_t,
      int4x8,
      scalar_t,
      mat_b_layout,
      mat_b_layout,
      mem_layout::row_major,
      mem_layout::row_major,
      q_info,
      int4_dequantize_attr,
      static_cast<gpu::xetla::gpu_arch>(ARCH)>;
  typename int4_dequantize_kernel::arguments_t args(
      k,
      n,
      reinterpret_cast<gpu::xetla::int4x8*>(const_cast<uint32_t*>(b)),
      const_cast<scalar_t*>(b_scale),
      reinterpret_cast<gpu::xetla::int4x8*>(const_cast<uint32_t*>(b_zp)),
      out,
      k,
      n,
      k / DEQUANT_S,
      n);
  cl::sycl::nd_range<3> nd_range = int4_dequantize_kernel::get_nd_range(args);
  XetlaInt4DequantizeRunFunctor<int4_dequantize_kernel> kfn(args);
  return [=](sycl::handler& cgh) {
    cgh.parallel_for<decltype(kfn)>(nd_range, kfn);
  };
}
} // namespace torch_ipex::xpu::xetla
