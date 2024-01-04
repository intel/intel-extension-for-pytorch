#pragma once

#include <utils/DPCPP.h>
#include "../../GEMM_INT4.h"
#include "../xetla.h"
#include "epilogue_impl.h"

namespace xpu {
namespace xetla {

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
    uint32_t dequant_s,
    typename post_ops>
struct hgemm_wint4_pvc_func {
  using tile_shape = gpu::xetla::group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
  static constexpr uint32_t periodic_sync_interval = 1;
  static constexpr uint32_t prefetch_distance = 3;

  using mem_desc_a_t =
      mem_desc_t<dtype_a, mem_layout::row_major, mem_space::global>;
  using mem_desc_b_t =
      mem_desc_t<dtype_b, mem_layout::row_major, mem_space::global>;
  using mem_desc_c_t =
      mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>;

  using compute_attr = gpu::xetla::group::compute_attr_t<fp16, fp16, dtype_acc>;
  using perf_tuning_knob = gpu::xetla::group::
      perf_tuning_knob_t<sg_k, prefetch_distance, periodic_sync_interval>;

  using compute_policy = gpu::xetla::group::compute_policy_int4_dequantize_xmx<
      compute_attr,
      perf_tuning_knob,
      dtype_scale,
      dtype_zero_pt,
      dequant_s == 0 ? 131072 : dequant_s,
      gpu_arch::Xe>;
  using gemm_t = gpu::xetla::group::
      gemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::epilogue_policy_tile_op<post_ops, gpu_arch::Xe>,
      tile_shape,
      mem_desc_c_t>;

  static_assert(
      l3_kslicing == 1 || std::is_same<remove_const_t<dtype_c>, float>::value ||
          std::is_same<remove_const_t<dtype_c>, int>::value,
      "for l3_kslicing > 1, current we only support float or "
      "int for matC");

  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<gpu_arch::Xe>;
  using dispatch_policy = dispatch_policy_int4_dequantize_kslicing<
      group_swizzle,
      l3_kslicing,
      slm_kslicing>;
  using gemm_op_t =
      gpu::xetla::kernel::gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;

  using dtype_cnt = uint32_t;

  static const char* func_name() {
    return "hgemm_wint4_pvc_func";
  }

  static inline void run(
      sycl::queue& queue,
      dtype_a* A,
      dtype_b* B,
      dtype_c* C,
      uint32_t mat_m,
      uint32_t mat_n,
      uint32_t mat_k,
      dtype_zero_pt* zero_pt_ptr,
      dtype_scale* scale_ptr,
      typename epilogue_t::arguments_t epilogue_args = {}) {
    // allocate temp buffers for global split
    size_t size_acc = gemm_op_t::get_acc_buf_size(mat_m, mat_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(mat_m, mat_n);

    using data_type_acc = float; // half * half  = float
    using data_type_cnt = uint32_t;
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();
    dtype_acc* acc = static_cast<data_type_acc*>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_acc * sizeof(dtype_acc), device, context));
    dtype_cnt* cnt = static_cast<uint32_t*>(aligned_alloc_device(
        DEVICE_MEM_ALIGNMENT, size_cnt * sizeof(dtype_cnt), device, context));

    typename gemm_op_t::arguments_t gemm_arg(
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
        acc,
        cnt,
        epilogue_args);

    cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

    auto cgf = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for(NDRange, [=](nd_item<3> item) KERNEL_MAIN {
        slm_barrier_init<gemm_op_t>();
        gemm_op_t gemm_op;
        gemm_op(item, gemm_arg);
      });
    };
    DPCPP_Q_SUBMIT(queue, cgf);
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
    int STAGES>
inline void hgemm_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      subgroup::chained_tile_op_t<>>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);
  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale));
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
    int STAGES>
inline void hgemm_bias_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<mem_desc_t<data_type_bias, mem_layout::row_major, mem_space::global>, gpu_arch::Xe>>;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;
  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}}}});
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
    int STAGES>
inline void hgemm_bias_gelu_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<mem_desc_t<data_type_bias, mem_layout::row_major, mem_space::global>, gpu_arch::Xe>,
      subgroup::gelu_fwd_op_t>;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}}, {}}});
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
    int STAGES>
inline void hgemm_res_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* res,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_res = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::
          elemwise_reduce_op_t<reduce_op::sum, data_type_res, gpu_arch::Xe>>;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(res), {n, m, n}}}});
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
    int STAGES>
inline void hgemm_mul_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* mul,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_mul = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::
          elemwise_reduce_op_t<reduce_op::prod, data_type_mul, gpu_arch::Xe>>;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(mul), {n, m, n}}}});
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
    int STAGES>
inline void hgemm_silu_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using post_op = subgroup::chained_tile_op_t<
      epilogue_impl::silu_op_t>;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{}}});
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
    int STAGES>
inline void hgemm_silu_mul_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* mul,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_mul = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      epilogue_impl::silu_op_t,
      subgroup::elemwise_reduce_op_t<reduce_op::prod, data_type_mul, gpu_arch::Xe>
      >;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{}, {const_cast<scalar_t*>(mul), {n, m, n}}}});
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
    int STAGES>
inline void hgemm_bias_silu_mul_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* mul,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_mul = scalar_t;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<mem_desc_t<data_type_bias, mem_layout::row_major, mem_space::global>, gpu_arch::Xe>,
      epilogue_impl::silu_op_t,
      subgroup::elemwise_reduce_op_t<reduce_op::prod, data_type_mul, gpu_arch::Xe>
      >;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}}, {}, {const_cast<scalar_t*>(mul), {n, m, n}}}});
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
    int STAGES>
inline void hgemm_bias_add_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* res,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_res = scalar_t;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<mem_desc_t<data_type_bias, mem_layout::row_major, mem_space::global>, gpu_arch::Xe>,
      subgroup::elemwise_reduce_op_t<reduce_op::sum, data_type_res, gpu_arch::Xe>
      >;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
  const data_type_zp* b_zp_alias = reinterpret_cast<const data_type_zp*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}}, {const_cast<scalar_t*>(res), {n, m, n}}}});
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
    int STAGES>
inline void hgemm_bias_res_res_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const scalar_t* res0,
    const scalar_t* res1,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_bias = scalar_t;
  using data_type_res = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::
          bias_add_op_t<mem_desc_t<data_type_bias, mem_layout::row_major, mem_space::global>, gpu_arch::Xe>,
      subgroup::
          elemwise_reduce_op_t<reduce_op::sum, data_type_res, gpu_arch::Xe>,
      subgroup::
          elemwise_reduce_op_t<reduce_op::sum, data_type_res, gpu_arch::Xe>>;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const int4x2*>(b);
  const data_type_b* b_zp_alias = reinterpret_cast<const int4x2*>(b_zp);

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}},
        {const_cast<scalar_t*>(res0), {n, m, n}},
        {const_cast<scalar_t*>(res1), {n, m, n}}}});
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
    int STAGES>
inline void hgemm_qkv_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");

  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using post_op = subgroup::chained_tile_op_t<>;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const int4x2*>(b);
  const data_type_b* b_zp_alias = reinterpret_cast<const int4x2*>(b_zp);

  uint32_t weight_offset = k * n / 2;

  uint32_t group_num = 1;
  if constexpr (DQUANT_S != 0) {
    group_num = k / DQUANT_S;
  }
  uint32_t zp_offset = group_num * n / 2;
  uint32_t scale_offset = group_num * n;

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out0,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale));
  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias + weight_offset),
      out1,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias + zp_offset),
      const_cast<scalar_t*>(b_scale + scale_offset));
  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias + 2 * weight_offset),
      out2,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias + 2 * zp_offset),
      const_cast<scalar_t*>(b_scale + 2 * scale_offset));
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
    int STAGES>
inline void hgemm_qkv_bias_wint4_pvc(
    sycl::queue& queue,
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const uint8_t* b,
    const uint8_t* b_zp,
    const scalar_t* b_scale,
    const scalar_t* bias,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  using data_type_a = scalar_t;
  using data_type_b = int4x2;
  using data_type_c = scalar_t;
  using data_type_zp = int4x2;
  using data_type_scale = scalar_t;
  using data_type_acc = float;
  using data_type_bias = scalar_t;
  using post_op = subgroup::chained_tile_op_t<
      subgroup::bias_add_op_t<
      mem_desc_t<data_type_bias, mem_layout::row_major, mem_space::global>, gpu_arch::Xe>>;
  using hgemm_wint4_pvc_functor = hgemm_wint4_pvc_func<
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
      DQUANT_S,
      post_op>;

  const data_type_b* b_alias = reinterpret_cast<const int4x2*>(b);
  const data_type_b* b_zp_alias = reinterpret_cast<const int4x2*>(b_zp);

  uint32_t weight_offset = k * n / 2;
  uint32_t bias_offset = k * n / 2;

  uint32_t group_num = 1;
  if constexpr (DQUANT_S != 0) {
    group_num = k / DQUANT_S;
  }
  uint32_t zp_offset = group_num * n / 2;
  uint32_t scale_offset = group_num * n;

  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias),
      out0,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias),
      const_cast<scalar_t*>(b_scale),
      {{{const_cast<scalar_t*>(bias), {n, 1, n}}}});
  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias + weight_offset),
      out1,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias + zp_offset),
      const_cast<scalar_t*>(b_scale + scale_offset),
      {{{const_cast<scalar_t*>(bias + n), {n, 1, n}}}});
  hgemm_wint4_pvc_functor::run(
      queue,
      const_cast<scalar_t*>(a),
      const_cast<data_type_b*>(b_alias + 2 * weight_offset),
      out2,
      m,
      n,
      k,
      const_cast<data_type_zp*>(b_zp_alias + 2 * zp_offset),
      const_cast<scalar_t*>(b_scale + 2 * scale_offset),
      {{{const_cast<scalar_t*>(bias + 2 * n), {n, 1, n}}}});
}



} // namespace xetla
} // namespace xpu
