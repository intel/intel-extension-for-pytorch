#pragma once

#include "../../GEMM_INT4.h"
#include "../xetla.h"
#include "epilogue_impl.h"
#include "mlp_gate_mul_up_int4_fwd.hpp"

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
      mem_layout::col_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_b)>;
  using mem_desc_c_t = mem_desc_t<
      dtype_c,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_c)>;

  using compute_attr = gpu::xetla::group::compute_attr_t<fp16, fp16, dtype_acc>;
  using perf_tuning_knob = gpu::xetla::group::
      perf_tuning_knob_t<sg_k, prefetch_distance, periodic_sync_interval>;

  static constexpr mma_engine mma_eng =
      (static_cast<gpu_arch>(arch) == gpu_arch::XeLpg || sg_m == 1)
      ? mma_engine::fpu
      : mma_engine::xmx;

  static constexpr quant_info quant_info{
      quant_mode::I4_SYM,
      dequant_s == 0 ? 131072 : dequant_s,
      mem_layout::col_major};

  using compute_policy = gpu::xetla::group::compute_policy_int4_dequantize<
      compute_attr,
      perf_tuning_knob,
      dtype_scale,
      dtype_zero_pt,
      quant_info,
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
  using op_args_t =
      typename gemm_op_t::template arguments_t<compute_policy::quant_mode>;

  template <typename gemm_op_t>
  struct GEMMFunctor {
    op_args_t<gemm_op_t> gemm_arg;

    KERNEL_MAIN void operator()(nd_item<3> item) const {
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(item, gemm_arg);
    }
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
    op_args_t<gemm_op_t> gemm_arg(
        mat_m,
        mat_k,
        mat_n,
        A,
        mat_k, // mat_k
        B,
        mat_k,
        C,
        mat_n,
        scale_ptr,
        dequant_s == 0 ? 1 : (mat_k / compute_policy::dequant_s),
        acc_ptr,
        cnt_ptr,
        epilogue_args);
    sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

    GEMMFunctor<gemm_op_t> kfn(gemm_arg);
    return [=](sycl::handler& cgh) {
      cgh.parallel_for<decltype(kfn)>(NDRange, kfn);
    };
  }

  template <typename gemm_op_t>
  struct QKVFunctor {
    op_args_t<gemm_op_t> gemm_arg;
    std::array<size_t, 3> NTile_offset;
    std::array<uint32_t, 3> C_ld;
    std::array<dtype_c*, 3> C_ptr;

    KERNEL_MAIN void operator()(nd_item<3> item) const {
      slm_barrier_init<gemm_op_t>();
      auto ntile_idx = item.get_global_id(2); // K, M, [N]
      size_t qkv012 = // 0: q, 1: k, 2: v
          ntile_idx < NTile_offset[1]   ? 0
          : ntile_idx < NTile_offset[2] ? 1
                                        : 2;
      op_args_t<gemm_op_t> item_args = gemm_arg;
      item_args.matC_ld = C_ld[qkv012];
      item_args.matC_base = C_ptr[qkv012];
      gemm_op_t gemm_op;
      gemm_op(item, item_args);
    }
  };
  static inline cgf_t run_qkv(
      dtype_a* A,
      dtype_b* B,
      // offset_n0 assumes to be 0
      dtype_c* C0,
      uint32_t ldc0,
      uint32_t offset_n1,
      dtype_c* C1,
      uint32_t ldc1,
      uint32_t offset_n2,
      dtype_c* C2,
      uint32_t ldc2,
      dtype_acc* acc_ptr,
      uint32_t* cnt_ptr,
      uint32_t mat_m,
      uint32_t mat_n,
      uint32_t mat_k,
      dtype_zero_pt* zero_pt_ptr,
      dtype_scale* scale_ptr,
      typename epilogue_t::arguments_t epilogue_args = {}) {
    assert(offset_n1 % sg_n == 0);
    assert(offset_n2 % sg_n == 0);
    assert(mat_n % sg_n == 0);
    op_args_t<gemm_op_t> gemm_arg(
        mat_m,
        mat_k,
        mat_n,
        A,
        mat_k, // mat_k
        B,
        mat_k,
        C0,
        mat_n,
        scale_ptr,
        dequant_s == 0 ? 1 : (mat_k / compute_policy::dequant_s),
        acc_ptr,
        cnt_ptr,
        epilogue_args);
    sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);
    QKVFunctor<gemm_op_t> kfn{
        gemm_arg,
        {0, offset_n1 / sg_n, offset_n2 / sg_n},
        {ldc0, ldc1, ldc2},
        {C0, C1 - offset_n1, C2 - offset_n2}};
    return [=](sycl::handler& cgh) {
      cgh.parallel_for<decltype(kfn)>(NDRange, kfn);
    };
  }
};

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
    uint32_t dequant_s,
    uint32_t periodic_sync_interval,
    uint32_t prefetch_distance,
    typename post_ops_gate_t,
    typename post_ops_up_t>
struct hgemm_mlp_gate_mul_up_wint4_func {
  using tile_shape = gpu::xetla::group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;

  using mem_desc_a_t = mem_desc_t<
      dtype_a,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_a)>;
  using mem_desc_b_t = mem_desc_t<
      dtype_b,
      mem_layout::col_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_b)>;
  using mem_desc_c_t = mem_desc_t<
      dtype_c,
      mem_layout::row_major,
      mem_space::global,
      DEVICE_MEM_ALIGNMENT / sizeof(dtype_c)>;

  using compute_attr = gpu::xetla::group::compute_attr_t<fp16, fp16, dtype_acc>;
  using perf_tuning_knob = gpu::xetla::group::
      perf_tuning_knob_t<sg_k, prefetch_distance, periodic_sync_interval>;

  static constexpr mma_engine mma_eng =
      (static_cast<gpu_arch>(arch) == gpu_arch::XeLpg || sg_m == 1)
      ? mma_engine::fpu
      : mma_engine::xmx;

  static constexpr quant_info quant_info{
      quant_mode::I4_SYM,
      dequant_s == 0 ? 131072 : dequant_s,
      mem_layout::col_major};

  using compute_policy = gpu::xetla::group::compute_policy_int4_dequantize<
      compute_attr,
      perf_tuning_knob,
      dtype_scale,
      dtype_zero_pt,
      quant_info,
      mma_eng,
      static_cast<gpu_arch>(arch)>;
  using gemm_t = gpu::xetla::group::
      gemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::epilogue_policy_default<static_cast<gpu_arch>(arch)>,
      tile_shape,
      mem_desc_c_t>;

  using mlp_op_t = mlp::mlp_gate_mul_up_int4_fwd_t<
      static_cast<gpu_arch>(arch),
      l3_kslicing,
      slm_kslicing,
      gemm_t,
      post_ops_up_t,
      post_ops_gate_t,
      epilogue_t>;

  static const char* func_name() {
    return "hgemm_mlp_gate_mul_up_wint4_func";
  }

  struct RunKernelFunctor {
    KERNEL_MAIN void operator()(nd_item<3> item) const {
      slm_barrier_init<mlp_op_t>();
      mlp_op_t gemm_op;
      gemm_op(item, gemm_arg);
    }
    RunKernelFunctor(typename mlp_op_t::arguments_t gemm_arg_)
        : gemm_arg(gemm_arg_) {}

   private:
    typename mlp_op_t::arguments_t gemm_arg;
  };

  static inline cgf_t run(
      dtype_a* A,
      dtype_b* B_gate,
      dtype_b* B_up,
      dtype_c* C,
      dtype_acc* acc_ptr,
      uint32_t* cnt_ptr,
      uint32_t mat_m,
      uint32_t mat_n,
      uint32_t mat_k,
      dtype_zero_pt* zp_gate,
      dtype_zero_pt* zp_up,
      dtype_scale* scale_gate,
      dtype_scale* scale_up,
      typename post_ops_gate_t::arguments_t post_ops_gate_args = {},
      typename post_ops_up_t::arguments_t post_ops_up_args = {}) {
    typename mlp_op_t::arguments_t mlp_arg(
        mat_m,
        mat_k,
        mat_n,
        A,
        mat_k,
        B_up,
        B_gate,
        mat_k,
        C,
        mat_n,
        acc_ptr + mlp_op_t::get_acc_buf_size(mat_m, mat_n),
        acc_ptr,
        cnt_ptr,
        {scale_up,
         scale_gate,
         zp_up,
         zp_gate,
         // scale_ld
         dequant_s == 0 ? 1 : (mat_k / compute_policy::dequant_s),
         // zp_ld
         dequant_s == 0 ? 1 : mat_n},
        post_ops_up_args,
        post_ops_gate_args);

    sycl::nd_range<3> NDRange = mlp_op_t::get_nd_range(mlp_arg);
    RunKernelFunctor kfn(mlp_arg);
    return [=](sycl::handler& cgh) {
      cgh.parallel_for<decltype(kfn)>(NDRange, kfn);
    };
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
    int ARCH>
inline cgfs_t hgemm_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
    int ARCH>
inline cgfs_t hgemm_bias_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
    int ARCH>
inline cgfs_t hgemm_bias_gelu_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
    int ARCH>
inline cgfs_t hgemm_res_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
    int ARCH>
inline cgfs_t hgemm_mul_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
    int ARCH>
inline cgfs_t hgemm_bias_res_res_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
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
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const int4x8*>(b);
  const data_type_b* b_zp_alias = reinterpret_cast<const int4x8*>(b_zp);

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
    int ARCH>
inline cgfs_t hgemm_qkv_wint4(
    scalar_t* out0,
    uint32_t ld_out0,
    uint32_t offset_n1,
    scalar_t* out1,
    uint32_t ld_out1,
    uint32_t offset_n2,
    scalar_t* out2,
    uint32_t ld_out2,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const int4x8*>(b);
  const data_type_b* b_zp_alias = reinterpret_cast<const int4x8*>(b_zp);

  uint32_t weight_offset = k * n / 8;

  uint32_t group_num = 1;
  if constexpr (DQUANT_S != 0) {
    group_num = k / DQUANT_S;
  }
  uint32_t zp_offset = group_num * n / 8;
  uint32_t scale_offset = group_num * n;

  return {hgemm_wint4_functor::run_qkv(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out0,
      ld_out0,
      offset_n1,
      out1,
      ld_out1,
      offset_n2,
      out2,
      ld_out2,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale))};
}

template <
    bool has_bias_gate,
    bool has_bias_up,
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
    int ARCH>
inline cgfs_t hgemm_mlp_silu_mul_wint4_helper(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k,
    const scalar_t* bias_gate,
    const scalar_t* bias_up) {
  static_assert(L3_KS == 1, "for mlp fusion, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using bias_op_t = subgroup::bias_add_op_t<
      mem_desc_t<
          scalar_t,
          mem_layout::row_major,
          mem_space::global,
          DEVICE_MEM_ALIGNMENT / sizeof(scalar_t)>,
      static_cast<gpu_arch>(ARCH)>;
  using post_ops_gate_t = std::conditional_t<
      has_bias_gate,
      subgroup::chained_tile_op_t<bias_op_t, subgroup::silu_op_t>,
      subgroup::chained_tile_op_t<subgroup::silu_op_t>>;
  using post_ops_up_t = std::conditional_t<
      has_bias_up,
      subgroup::chained_tile_op_t<bias_op_t>,
      subgroup::chained_tile_op_t<>>;
  using hgemm_mlp_gate_mul_up_wint4_functor = hgemm_mlp_gate_mul_up_wint4_func<
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
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_ops_gate_t,
      post_ops_up_t>;
  typename post_ops_gate_t::arguments_t post_ops_gate_args;
  if constexpr (has_bias_gate) {
    post_ops_gate_args = {{const_cast<scalar_t*>(bias_gate), {n, 1, n}}, {}};
  }
  typename post_ops_up_t::arguments_t post_ops_up_args;
  if constexpr (has_bias_up) {
    post_ops_up_args = {{const_cast<scalar_t*>(bias_up), {n, 1, n}}};
  }

  const data_type_b* b_alias = reinterpret_cast<const int4x8*>(b);
  const data_type_b* b_zp_alias = reinterpret_cast<const int4x8*>(b_zp);

  uint32_t group_num = 1;
  if constexpr (DQUANT_S != 0) {
    group_num = k / DQUANT_S;
  }
  uint32_t b_offset = k * n / 8;
  uint32_t zp_offset = group_num * n / 8;
  uint32_t scale_offset = group_num * n;
  return {hgemm_mlp_gate_mul_up_wint4_functor::run(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      const_cast<data_type_b*>(b_alias + b_offset),
      out,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<data_type_zp*>(
          b_zp_alias == nullptr ? nullptr : b_zp_alias + zp_offset),
      const_cast<scalar_t*>(b_scale),
      const_cast<scalar_t*>(b_scale + scale_offset),
      post_ops_gate_args,
      post_ops_up_args)};
}

template <typename scalar_t, int... CONFIG_ARGS>
inline cgfs_t hgemm_mlp_silu_mul_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return hgemm_mlp_silu_mul_wint4_helper<
      false,
      false,
      scalar_t,
      CONFIG_ARGS...>(
      out, a, b, b_zp, b_scale, acc_ptr, cnt_ptr, m, n, k, nullptr, nullptr);
}
template <typename scalar_t, int... CONFIG_ARGS>
inline cgfs_t hgemm_mlp_bias_silu_mul_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias_gate,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return hgemm_mlp_silu_mul_wint4_helper<true, false, scalar_t, CONFIG_ARGS...>(
      out, a, b, b_zp, b_scale, acc_ptr, cnt_ptr, m, n, k, bias_gate, nullptr);
}
template <typename scalar_t, int... CONFIG_ARGS>
inline cgfs_t hgemm_mlp_silu_mul_bias_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias_up,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return hgemm_mlp_silu_mul_wint4_helper<false, true, scalar_t, CONFIG_ARGS...>(
      out, a, b, b_zp, b_scale, acc_ptr, cnt_ptr, m, n, k, nullptr, bias_up);
}
template <typename scalar_t, int... CONFIG_ARGS>
inline cgfs_t hgemm_mlp_bias_silu_mul_bias_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias_gate,
    const scalar_t* bias_up,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  return hgemm_mlp_silu_mul_wint4_helper<true, true, scalar_t, CONFIG_ARGS...>(
      out, a, b, b_zp, b_scale, acc_ptr, cnt_ptr, m, n, k, bias_gate, bias_up);
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
    int ARCH>
inline cgfs_t hgemm_qkv_bias_wint4(
    scalar_t* out0,
    uint32_t ld_out0,
    uint32_t offset_n1,
    scalar_t* out1,
    uint32_t ld_out1,
    uint32_t offset_n2,
    scalar_t* out2,
    uint32_t ld_out2,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
      DQUANT_S,
      SYNC_FREQ,
      STAGES,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const int4x8*>(b);
  const data_type_b* b_zp_alias = reinterpret_cast<const int4x8*>(b_zp);

  uint32_t weight_offset = k * n / 8;
  uint32_t bias_offset = k * n / 8;

  uint32_t group_num = 1;
  if constexpr (DQUANT_S != 0) {
    group_num = k / DQUANT_S;
  }
  uint32_t zp_offset = group_num * n / 8;
  uint32_t scale_offset = group_num * n;
  return {hgemm_wint4_functor::run_qkv(
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out0,
      ld_out0,
      offset_n1,
      out1,
      ld_out1,
      offset_n2,
      out2,
      ld_out2,
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
    int ARCH>
inline cgfs_t hgemm_silu_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using post_op = subgroup::chained_tile_op_t<subgroup::silu_op_t>;
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
    int ARCH>
inline cgfs_t hgemm_silu_mul_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_mul = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::silu_op_t,
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
    int ARCH>
inline cgfs_t hgemm_bias_silu_mul_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
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
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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
      subgroup::silu_op_t,
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
    int ARCH>
inline cgfs_t hgemm_bias_add_wint4(
    scalar_t* out,
    const scalar_t* a,
    const uint32_t* b,
    const uint32_t* b_zp,
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
  using data_type_b = int4x8;
  using data_type_c = scalar_t;
  using data_type_zp = int4x8;
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

} // namespace torch_ipex::xpu::xetla
