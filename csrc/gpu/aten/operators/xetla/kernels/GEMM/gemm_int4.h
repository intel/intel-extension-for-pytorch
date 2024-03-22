#pragma once

#include "../../GEMM_INT4.h"
#include "../xetla.h"
#include "../xetlaDispatch.h"
#include "epilogue_impl.h"

namespace torch_ipex::xpu::xetla {
using namespace gpu::xetla;

template <
    typename dtype_a,
    typename dtype_b,
    typename dtype_c,
    typename dtype_zero_pt,
    typename dtype_scale,
    typename dtype_acc,
    uint32_t wg_m,
    uint32_t wg_n,
    uint32_t sg_m,
    uint32_t sg_n,
    uint32_t sg_k,
    uint32_t l3_kslicing,
    uint32_t slm_kslicing,
    uint32_t arch,
    uint32_t q_mode,
    uint32_t dequant_s,
    uint32_t periodic_sync_interval,
    uint32_t prefetch_distance,
    typename post_ops>
struct hgemm_wint4_func {
  using tile_shape = gpu::xetla::group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;

  using mem_desc_a_t = mem_desc_t<
      dtype_a,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_a)>;
  using mem_desc_b_t = mem_desc_t<
      dtype_b,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_b)>;
  using mem_desc_c_t = mem_desc_t<
      dtype_c,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_c)>;
  using mem_desc_scale_t = mem_desc_t<
      dtype_scale,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_scale)>;

  using compute_attr =
      gpu::xetla::group::compute_attr_t<dtype_a, dtype_a, dtype_acc>;
  using perf_tuning_knob = gpu::xetla::group::
      perf_tuning_knob_t<sg_k, prefetch_distance, periodic_sync_interval>;

  static constexpr mma_engine mma_eng =
      (static_cast<gpu_arch>(arch) == gpu_arch::XeLpg) ? mma_engine::fpu
                                                       : mma_engine::xmx;
  using compute_policy = gpu::xetla::group::compute_policy_int4_dequantize<
      compute_attr,
      perf_tuning_knob,
      dtype_scale,
      dtype_zero_pt,
      static_cast<gpu::xetla::group::quant_mode>(q_mode),
      dequant_s == 0 ? 131072 : dequant_s,
      mma_eng,
      static_cast<gpu_arch>(arch)>;
  using gemm_t = gpu::xetla::group::
      gemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::
          epilogue_policy_tile_op<post_ops, static_cast<gpu_arch>(arch)>,
      tile_shape,
      mem_desc_c_t>;

  using group_swizzle =
      gpu::xetla::kernel::group_swizzle_default<static_cast<gpu_arch>(arch)>;
  using dispatch_policy = dispatch_policy_int4_dequantize_kslicing<
      group_swizzle,
      l3_kslicing,
      slm_kslicing>;
  using gemm_op_t =
      gpu::xetla::kernel::gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;

  static const char* func_name() {
    return "hgemm_wint4_func";
  }

  template <typename gemm_op_t>
  struct RunKernelFunctor {
    KERNEL_MAIN void operator()(nd_item<3> item) const {
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(item, gemm_arg);
    }
    RunKernelFunctor(
        typename gemm_op_t::template arguments_t<compute_policy::quant_type>
            gemm_arg_)
        : gemm_arg(gemm_arg_) {}

   private:
    typename gemm_op_t::template arguments_t<compute_policy::quant_type>
        gemm_arg;
  };

  static inline cgf_t run(
      dtype_a* A,
      dtype_b* B,
      dtype_c* C,
      dtype_acc* acc_ptr,
      uint32_t* cnt_ptr,
      uint32_t mat_m,
      uint32_t mat_n,
      uint32_t mat_k,
      dtype_zero_pt* zero_pt_ptr,
      dtype_scale* scale_ptr,
      typename epilogue_t::arguments_t epilogue_args = {}) {
    if constexpr (
        static_cast<gpu::xetla::group::quant_mode>(q_mode) ==
        gpu::xetla::group::quant_mode::S4_FULLRANGE_NO_ZP) {
      typename gemm_op_t::template arguments_t<compute_policy::quant_type>
          gemm_arg(
              mat_m,
              mat_k,
              mat_n,
              A,
              mat_k,
              B,
              mat_n,
              C,
              mat_n,
              scale_ptr,
              mat_n,
              acc_ptr,
              cnt_ptr,
              epilogue_args);

      cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

      RunKernelFunctor<gemm_op_t> kfn(gemm_arg);
      return [=](sycl::handler& cgh) {
        cgh.parallel_for<decltype(kfn)>(NDRange, kfn);
      };
    } else {
      typename gemm_op_t::template arguments_t<compute_policy::quant_type>
          gemm_arg(
              mat_m,
              mat_k,
              mat_n,
              A,
              mat_k,
              B,
              mat_n,
              C,
              mat_n,
              scale_ptr,
              mat_n,
              zero_pt_ptr,
              mat_n,
              acc_ptr,
              cnt_ptr,
              epilogue_args);

      cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

      RunKernelFunctor<gemm_op_t> kfn(gemm_arg);
      return [=](sycl::handler& cgh) {
        cgh.parallel_for<decltype(kfn)>(NDRange, kfn);
      };
    }
  }
};

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      subgroup::chained_tile_op_t<>>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);
  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale))};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_bias = scalar_t;
  using mem_desc_bias_t = mem_desc_t<
      data_type_bias,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(data_type_bias)>;
  using bias_op_t =
      subgroup::bias_add_op_t<mem_desc_bias_t, static_cast<gpu_arch>(ARCH)>;
  using post_op = subgroup::chained_tile_op_t<bias_op_t>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;
  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}}}})};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_gelu_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<
          mem_desc_t<
              data_type_bias,
              mem_layout::row_major,
              mem_space::global,
              DEVICE_MEM_ALIGNMENT / sizeof(data_type_bias)>,
          static_cast<gpu_arch>(ARCH)>,
      subgroup::gelu_fwd_op_t>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}}, {}}})};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_res_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_res = scalar_t;
  using post_op = subgroup::chained_tile_op_t<subgroup::elemwise_reduce_op_t<
      reduce_op::sum,
      data_type_res,
      static_cast<gpu_arch>(ARCH)>>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(res), {n, m, n}}}})};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_mul_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_mul = scalar_t;
  using post_op = subgroup::chained_tile_op_t<subgroup::elemwise_reduce_op_t<
      reduce_op::prod,
      data_type_mul,
      static_cast<gpu_arch>(ARCH)>>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(mul), {n, m, n}}}})};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_res_res_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* res0,
    const scalar_t* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_bias = scalar_t;
  using data_type_res = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<
          mem_desc_t<
              data_type_bias,
              mem_layout::row_major,
              mem_space::global,
              DEVICE_MEM_ALIGNMENT / sizeof(data_type_bias)>,
          static_cast<gpu_arch>(ARCH)>,
      subgroup::elemwise_reduce_op_t<
          reduce_op::sum,
          data_type_res,
          static_cast<gpu_arch>(ARCH)>,
      subgroup::elemwise_reduce_op_t<
          reduce_op::sum,
          data_type_res,
          static_cast<gpu_arch>(ARCH)>>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}},
        {const_cast<scalar_t*>(res0), {n, m, n}},
        {const_cast<scalar_t*>(res1), {n, m, n}}}})};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_qkv_wint4(
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using post_op = subgroup::chained_tile_op_t<>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  uint32_t weight_offset = k * n / 2;

  uint32_t group_num = 1;
  if constexpr (DQUANT_S != 0) {
    group_num = k / DQUANT_S;
  }
  uint32_t zp_offset = group_num * n / 2;
  uint32_t scale_offset = group_num * n;
  return {

      hgemm_wint4_functor::run(
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias),
          out0,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          const_cast<data_type_zp*>(b_zp_alias),
          const_cast<scalar_t*>(b_scale)),
      hgemm_wint4_functor::run(
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias + weight_offset),
          out1,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          const_cast<data_type_zp*>(b_zp_alias + zp_offset),
          const_cast<scalar_t*>(b_scale + scale_offset)),
      hgemm_wint4_functor::run(
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias + 2 * weight_offset),
          out2,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          const_cast<data_type_zp*>(b_zp_alias + 2 * zp_offset),
          const_cast<scalar_t*>(b_scale + 2 * scale_offset)),
  };
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_qkv_bias_wint4(
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<subgroup::bias_add_op_t<
      mem_desc_t<
          data_type_bias,
          mem_layout::row_major,
          mem_space::global,
          DEVICE_MEM_ALIGNMENT / sizeof(data_type_bias)>,
      static_cast<gpu_arch>(ARCH)>>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  uint32_t weight_offset = k * n / 2;
  uint32_t bias_offset = k * n / 2;

  uint32_t group_num = 1;
  if constexpr (DQUANT_S != 0) {
    group_num = k / DQUANT_S;
  }
  uint32_t zp_offset = group_num * n / 2;
  uint32_t scale_offset = group_num * n;
  return {
      hgemm_wint4_functor::run(
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias),
          out0,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          const_cast<data_type_zp*>(b_zp_alias),
          const_cast<scalar_t*>(b_scale),
          {{{const_cast<scalar_t*>(bias), {n, 1, n}}}}),
      hgemm_wint4_functor::run(
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias + weight_offset),
          out1,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          const_cast<data_type_zp*>(b_zp_alias + zp_offset),
          const_cast<scalar_t*>(b_scale + scale_offset),
          {{{const_cast<scalar_t*>(bias + n), {n, 1, n}}}}),
      hgemm_wint4_functor::run(
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias + 2 * weight_offset),
          out2,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          const_cast<data_type_zp*>(b_zp_alias + 2 * zp_offset),
          const_cast<scalar_t*>(b_scale + 2 * scale_offset),
          {{{const_cast<scalar_t*>(bias + 2 * n), {n, 1, n}}}}),
  };
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_silu_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using post_op = subgroup::chained_tile_op_t<epilogue_impl::silu_op_t>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;
  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{}}})};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_silu_mul_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_mul = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      epilogue_impl::silu_op_t,
      subgroup::elemwise_reduce_op_t<
          reduce_op::prod,
          data_type_mul,
          static_cast<gpu_arch>(ARCH)>>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);
  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{}, {const_cast<scalar_t*>(mul), {n, m, n}}}})};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_silu_mul_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_mul = scalar_t;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<
          mem_desc_t<
              data_type_bias,
              mem_layout::row_major,
              mem_space::global,
              DEVICE_MEM_ALIGNMENT / sizeof(data_type_bias)>,
          static_cast<gpu_arch>(ARCH)>,
      epilogue_impl::silu_op_t,
      subgroup::elemwise_reduce_op_t<
          reduce_op::prod,
          data_type_mul,
          static_cast<gpu_arch>(ARCH)>>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);
  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}},
        {},
        {const_cast<scalar_t*>(mul), {n, m, n}}}})};
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_add_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    void* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = std::conditional_t<
      QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD),
      scalar_t,
      int4x2>;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_res = scalar_t;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<
          mem_desc_t<
              data_type_bias,
              mem_layout::row_major,
              mem_space::global,
              DEVICE_MEM_ALIGNMENT / sizeof(data_type_bias)>,
          static_cast<gpu_arch>(ARCH)>,
      subgroup::elemwise_reduce_op_t<
          reduce_op::sum,
          data_type_res,
          static_cast<gpu_arch>(ARCH)>>;
  using hgemm_wint4_functor = hgemm_wint4_func<
      data_type_a,
      data_type_b,
      data_type_c,
      data_type_zp,
      data_type_scale,
      data_type_acc,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      L3_KS,
      SLM_KS,
      ARCH,
      QUANT_MODE,
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  return {hgemm_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}},
        {const_cast<scalar_t*>(res), {n, m, n}}}})};
}

template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_bias_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(bias),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_gelu_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_bias_gelu_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(bias),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_mul_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_mul_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(mul),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_silu_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_silu_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_res_res_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    void* res0,
    void* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_bias_res_res_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(bias),
          reinterpret_cast<scalar_t*>(res0),
          reinterpret_cast<scalar_t*>(res1),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_qkv_wint4(
    XetlaType xe_type,
    void* out0,
    void* out1,
    void* out2,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_qkv_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out0),
          reinterpret_cast<scalar_t*>(out1),
          reinterpret_cast<scalar_t*>(out2),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_qkv_bias_wint4(
    XetlaType xe_type,
    void* out0,
    void* out1,
    void* out2,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_qkv_bias_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out0),
          reinterpret_cast<scalar_t*>(out1),
          reinterpret_cast<scalar_t*>(out2),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(bias),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_silu_mul_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_silu_mul_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(mul),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_silu_mul_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    void* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_bias_silu_mul_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(bias),
          reinterpret_cast<scalar_t*>(mul),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_bias_add_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* bias,
    void* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_bias_add_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(bias),
          reinterpret_cast<scalar_t*>(res),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}
template <
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int DQUANT_S,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    int ARCH,
    int QUANT_MODE>
inline cgfs_t hgemm_res_wint4(
    XetlaType xe_type,
    void* out,
    void* a,
    const uint8_t* b,
    void* b_zp,
    void* b_scale,
    void* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return XETLA_ALL_KERNEL_TYPES(xe_type, [&] {
    if constexpr (
        ARCH == static_cast<int>(gpu_arch::XeLpg) &&
        (std::is_same<scalar_t, gpu::xetla::bf16>::value ||
         QUANT_MODE == static_cast<int>(quant_mode::S4_ASYM_ZERO_NO_DEGRAD))) {
      std::cerr
          << "XeLpg not support bf16 type with S4_ASYM_ZERO_NO_DEGRAD mode!"
          << std::endl;
      return cgfs_t();
    } else {
      return hgemm_res_wint4<
          scalar_t,
          WG_M,
          WG_N,
          SG_M,
          SG_N,
          SG_K,
          DQUANT_S,
          SLM_KS,
          L3_KS,
          SYNC_FREQ,
          STAGES,
          ARCH,
          QUANT_MODE>(
          reinterpret_cast<scalar_t*>(out),
          reinterpret_cast<scalar_t*>(a),
          b,
          b_zp,
          reinterpret_cast<scalar_t*>(b_scale),
          reinterpret_cast<scalar_t*>(res),
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k);
    }
  });
}

#define HGEMM_WINT4_INTERFACE_IMPL(          \
    WG_M,                                    \
    WG_N,                                    \
    SG_M,                                    \
    SG_N,                                    \
    SG_K,                                    \
    DQUANT_S,                                \
    SLM_KS,                                  \
    L3_KS,                                   \
    SYNC_FREQ,                               \
    STAGES,                                  \
    ARCH,                                    \
    QUANT_MODE)                              \
  template cgfs_t hgemm_wint4<               \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_bias_wint4<          \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* bias,                            \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_bias_gelu_wint4<     \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* bias,                            \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_mul_wint4<           \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* mul,                             \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_silu_wint4<          \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_bias_res_res_wint4<  \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* bias,                            \
      void* res0,                            \
      void* res1,                            \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_qkv_wint4<           \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out0,                            \
      void* out1,                            \
      void* out2,                            \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_qkv_bias_wint4<      \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out0,                            \
      void* out1,                            \
      void* out2,                            \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* bias,                            \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_silu_mul_wint4<      \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* mul,                             \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_bias_silu_mul_wint4< \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* bias,                            \
      void* mul,                             \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_bias_add_wint4<      \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* bias,                            \
      void* res,                             \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);                     \
  template cgfs_t hgemm_res_wint4<           \
      WG_M,                                  \
      WG_N,                                  \
      SG_M,                                  \
      SG_N,                                  \
      SG_K,                                  \
      DQUANT_S,                              \
      SLM_KS,                                \
      L3_KS,                                 \
      SYNC_FREQ,                             \
      STAGES,                                \
      ARCH,                                  \
      QUANT_MODE>(                           \
      XetlaType xe_type,                     \
      void* out,                             \
      void* a,                               \
      const uint8_t* b,                      \
      void* b_zp,                            \
      void* b_scale,                         \
      void* res,                             \
      float* acc_ptr,                        \
      uint32_t* cnt_ptr,                     \
      const uint32_t m,                      \
      const uint32_t n,                      \
      const uint32_t k);
} // namespace torch_ipex::xpu::xetla
