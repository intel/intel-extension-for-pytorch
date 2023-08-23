#pragma once

#include <utils/DPCPP.h>
#include "../xetla.h"

namespace xpu {
namespace xetla {

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
    typename post_ops,
    bool noPostOp = false>
struct hgemm_wint4_func {
  using tile_shape = gpu::xetla::group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
  static constexpr uint32_t periodic_sync_interval = 1;
  static constexpr uint32_t prefetch_distance = 3;

  using dtype_table = fp16;

  using mem_desc_a_t =
      mem_desc_t<dtype_a, mem_layout::row_major, mem_space::global>;
  using mem_desc_b_t =
      mem_desc_t<dtype_b, mem_layout::row_major, mem_space::global>;
  using mem_desc_c_t =
      mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>;
  using mem_desc_scale_t =
      mem_desc_t<dtype_scale, mem_layout::row_major, mem_space::global>;
  using mem_desc_zero_pt_t =
      mem_desc_t<dtype_zero_pt, mem_layout::row_major, mem_space::global>;
  using mem_desc_table_t =
      mem_desc_t<dtype_table, mem_layout::row_major, mem_space::global>;

  using compute_attr = gpu::xetla::group::compute_attr_t<fp16, fp16, dtype_acc>;
  using perf_tuning_knob = gpu::xetla::group::
      perf_tuning_knob_t<sg_k, prefetch_distance, periodic_sync_interval>;

  using compute_policy = gpu::xetla::group::compute_policy_dequantize_matB_xmx<
      compute_attr,
      perf_tuning_knob,
      dtype_scale,
      dtype_zero_pt,
      dtype_table,
      dequant_s == 0 ? 131072 : dequant_s,
      gpu_arch::Xe>;
  using brgemm_t = gpu::xetla::group::
      brgemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using work_group_t = typename brgemm_t::work_group_t;
  static constexpr uint32_t work_group_size = work_group_t::size;
  using brgemm_args_t = typename brgemm_t::arguments_t;
  using matAcc_t = typename brgemm_t::matAcc_t;

  using update_method = typename std::
      conditional<(l3_kslicing > 1), result_reduce_sum, result_overwrite>::type;
  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::
          epilogue_policy_tile_op<post_ops, update_method, gpu_arch::Xe>,
      tile_shape,
      mem_desc_c_t>;
  using epilogue_args_t = typename epilogue_t::arguments_t;

  static_assert(
      l3_kslicing == 1 || std::is_same<remove_const_t<dtype_c>, float>::value ||
          std::is_same<remove_const_t<dtype_c>, int>::value,
      "for l3_kslicing > 1, current we only support float or "
      "int for matC");

  using kslicing_t = gpu::xetla::group::cooperative_reduce_t<
      reduce_op::sum,
      tile_shape,
      matAcc_t,
      slm_kslicing,
      gpu_arch::Xe>;
  using mat_slice_t = typename kslicing_t::mat_slice_t;

  static constexpr uint32_t brgemm_nbarr_count = brgemm_t::barrier_count;
  static constexpr uint32_t brgemm_slm_size = brgemm_t::slm_size;

  static constexpr uint32_t epilogue_nbarr_count = epilogue_t::barrier_count;
  static constexpr uint32_t epilogue_slm_size = epilogue_t::slm_size;

  static constexpr uint32_t kslicing_nbarr_count = kslicing_t::barrier_count;
  static constexpr uint32_t kslicing_slm_size = kslicing_t::slm_size;

  static constexpr uint32_t barrier_count = brgemm_nbarr_count * slm_kslicing +
      kslicing_nbarr_count + epilogue_nbarr_count * slm_kslicing;
  static constexpr uint32_t slm_size = brgemm_slm_size * slm_kslicing +
      kslicing_slm_size + epilogue_slm_size * slm_kslicing;

  static const char* func_name() {
    return "hgemm_wint4_func";
  }

  static inline void run(
      xetla_exec_item<3>& ei,
      dtype_a* A,
      dtype_b* B,
      dtype_c* C,
      uint32_t mat_m,
      uint32_t mat_n,
      uint32_t mat_k,
      uint32_t lda,
      uint32_t ldb,
      uint32_t ldc,
      dtype_zero_pt* zero_pt_ptr,
      dtype_scale* scale_ptr,
      uint32_t group_num,
      epilogue_args_t args = {}) {
    work_group_t g(ei.get_local_linear_id() % work_group_size);
    uint32_t slm_slice_id = ei.get_local_linear_id() / work_group_size;
    int start_n = ei.get_group(2) * wg_n;
    int start_m = ei.get_group(1) * wg_m;
    int start_k = 0;
    uint32_t wg_tile_k = mat_k;
    uint32_t boundary_k = wg_tile_k;
    if constexpr (l3_kslicing > 1) {
      wg_tile_k = (wg_tile_k + l3_kslicing - 1) / l3_kslicing;
      start_k = start_k + ei.get_group(0) * wg_tile_k;
      boundary_k = (start_k + wg_tile_k) > boundary_k ? boundary_k
                                                      : (start_k + wg_tile_k);
    }
    if constexpr (slm_kslicing > 1) {
      wg_tile_k = (wg_tile_k + slm_kslicing - 1) / slm_kslicing;
      start_k = start_k + slm_slice_id * wg_tile_k;
      boundary_k = (start_k + wg_tile_k) > boundary_k ? boundary_k
                                                      : (start_k + wg_tile_k);
    }

    int group_id = start_k / brgemm_t::dequant_s;

    uint32_t brgemm_slm_base = 0;
    uint32_t brgemm_nbarr_base = 0;
    if constexpr (slm_kslicing > 1) {
      brgemm_slm_base = slm_slice_id * brgemm_slm_size;
      brgemm_nbarr_base = slm_slice_id * brgemm_nbarr_count;
    }
    uint32_t kslicing_slm_base = slm_kslicing * brgemm_slm_size;
    uint32_t kslicing_nbarr_base = slm_kslicing * brgemm_nbarr_count;
    uint32_t epilogue_slm_base = kslicing_slm_base + kslicing_slm_size;
    uint32_t epilogue_nbarr_base = kslicing_nbarr_base + kslicing_nbarr_count;

    uint32_t inner_loop_count = (wg_tile_k + sg_k - 1) / sg_k;
    mem_desc_a_t mem_desc_a({A}, {boundary_k, mat_m, lda}, {start_k, start_m});
    mem_desc_b_t mem_desc_b(
        {B}, {mat_n / 2, boundary_k, ldb}, {start_n / 2, start_k});

    mem_desc_scale_t mem_desc_scale(
        {scale_ptr}, {mat_n, group_num, mat_n}, {start_n, group_id});

    mem_desc_zero_pt_t mem_desc_zero_pt(
        {zero_pt_ptr},
        {mat_n / 2, group_num, mat_n / 2},
        {start_n / 2, group_id});

    mem_desc_table_t mem_desc_table({nullptr}, {16, 1, 16}, {0, 0});

    matAcc_t matAcc;
    matAcc.init(0);
    brgemm_t brgemm;
    brgemm_args_t brgemm_args(
        mem_desc_a,
        mem_desc_b,
        inner_loop_count,
        mem_desc_scale,
        mem_desc_zero_pt,
        mem_desc_table);
    brgemm(g, matAcc, brgemm_args, brgemm_slm_base, brgemm_nbarr_base);

    kslicing_t kslicing(slm_slice_id);
    mat_slice_t mat_slice;
    kslicing(g, mat_slice, matAcc, kslicing_slm_base, kslicing_nbarr_base);

    int32_t coop_offset_n = kslicing.coop_id_x * mat_slice_t::tile_size_x;
    int32_t coop_offset_m = kslicing.coop_id_y * mat_slice_t::tile_size_y;
    mem_desc_c_t mem_desc_c(
        {C},
        {mat_n, mat_m, ldc},
        {start_n + coop_offset_n, start_m + coop_offset_m});

    if (!noPostOp) {
      epilogue_t epilogue;
      epilogue(
          g,
          mat_slice,
          mem_desc_c,
          args,
          epilogue_slm_base,
          epilogue_nbarr_base);
    }
  }
};

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3>
inline void hgemm_wint4(
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
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = n / 2;
  uint32_t ldc = n;
  cl::sycl::range<3> GroupRange{L3_KS, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);
      using data_type_a = scalar_t;
      using data_type_b = bit4x2;
      using data_type_c = scalar_t;
      using data_type_zp = bit4x2;
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
          DQUANT_S,
          subgroup::chained_tile_op_t<>,
          false>;
      constexpr uint32_t barrier_count = hgemm_wint4_functor::barrier_count;
      constexpr uint32_t slm_size = hgemm_wint4_functor::slm_size;

      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
      const data_type_zp* b_zp_alias =
          reinterpret_cast<const data_type_zp*>(b_zp);

      hgemm_wint4_functor::run(
          ei,
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias),
          out,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          const_cast<data_type_zp*>(b_zp_alias),
          const_cast<scalar_t*>(b_scale),
          DQUANT_S == 0 ? 1 : k / DQUANT_S,
          {});
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3>
inline void hgemm_bias_wint4(
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
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = n / 2;
  uint32_t ldc = n;
  cl::sycl::range<3> GroupRange{L3_KS, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);
      using data_type_a = scalar_t;
      using data_type_b = bit4x2;
      using data_type_c = scalar_t;
      using data_type_zp = bit4x2;
      using data_type_scale = scalar_t;
      using data_type_acc = float;
      using data_type_bias = scalar_t;
      using post_op = subgroup::chained_tile_op_t<
          subgroup::bias_add_op_t<data_type_bias, gpu_arch::Xe>>;
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
          DQUANT_S,
          post_op>;
      constexpr uint32_t barrier_count = hgemm_wint4_functor::barrier_count;
      constexpr uint32_t slm_size = hgemm_wint4_functor::slm_size;

      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
      const data_type_zp* b_zp_alias =
          reinterpret_cast<const data_type_zp*>(b_zp);

      hgemm_wint4_functor::run(
          ei,
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias),
          out,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          const_cast<data_type_zp*>(b_zp_alias),
          const_cast<scalar_t*>(b_scale),
          DQUANT_S == 0 ? 1 : k / DQUANT_S,
          {{{const_cast<scalar_t*>(bias), {n, 1, n}}}});
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3>
inline void hgemm_bias_gelu_wint4(
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
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = n / 2;
  uint32_t ldc = n;
  cl::sycl::range<3> GroupRange{L3_KS, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);
      using data_type_a = scalar_t;
      using data_type_b = bit4x2;
      using data_type_c = scalar_t;
      using data_type_zp = bit4x2;
      using data_type_scale = scalar_t;
      using data_type_acc = float;
      using data_type_bias = scalar_t;
      using post_op = subgroup::chained_tile_op_t<
          subgroup::bias_add_op_t<data_type_bias, gpu_arch::Xe>,
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
          DQUANT_S,
          post_op>;
      constexpr uint32_t barrier_count = hgemm_wint4_functor::barrier_count;
      constexpr uint32_t slm_size = hgemm_wint4_functor::slm_size;

      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
      const data_type_zp* b_zp_alias =
          reinterpret_cast<const data_type_zp*>(b_zp);

      hgemm_wint4_functor::run(
          ei,
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias),
          out,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          const_cast<data_type_zp*>(b_zp_alias),
          const_cast<scalar_t*>(b_scale),
          DQUANT_S == 0 ? 1 : k / DQUANT_S,
          {{{const_cast<scalar_t*>(bias), {n, 1, n}}, {}}});
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3>
inline void hgemm_res_wint4(
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
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = n / 2;
  uint32_t ldc = n;
  cl::sycl::range<3> GroupRange{L3_KS, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);
      using data_type_a = scalar_t;
      using data_type_b = bit4x2;
      using data_type_c = scalar_t;
      using data_type_zp = bit4x2;
      using data_type_scale = scalar_t;
      using data_type_acc = float;
      using data_type_res = scalar_t;
      using post_op =
          subgroup::chained_tile_op_t<subgroup::elemwise_reduce_op_t<
              reduce_op::sum,
              data_type_res,
              gpu_arch::Xe>>;
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
          DQUANT_S,
          post_op>;

      constexpr uint32_t barrier_count = hgemm_wint4_functor::barrier_count;
      constexpr uint32_t slm_size = hgemm_wint4_functor::slm_size;

      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      const data_type_b* b_alias = reinterpret_cast<const data_type_b*>(b);
      const data_type_zp* b_zp_alias =
          reinterpret_cast<const data_type_zp*>(b_zp);

      hgemm_wint4_functor::run(
          ei,
          const_cast<scalar_t*>(a),
          const_cast<data_type_b*>(b_alias),
          out,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          const_cast<data_type_zp*>(b_zp_alias),
          const_cast<scalar_t*>(b_scale),
          DQUANT_S == 0 ? 1 : k / DQUANT_S,
          {{{const_cast<scalar_t*>(res), {n, m, n}}}});
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3>
inline void hgemm_bias_res_res_wint4(
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
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = n / 2;
  uint32_t ldc = n;
  uint32_t ld_scale = n;
  cl::sycl::range<3> GroupRange{L3_KS, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);
      using data_type_a = scalar_t;
      using data_type_b = bit4x2;
      using data_type_c = scalar_t;
      using data_type_zp = bit4x2;
      using data_type_scale = scalar_t;
      using data_type_acc = float;
      using data_type_bias = scalar_t;
      using data_type_res = scalar_t;
      using post_op = subgroup::chained_tile_op_t<
          subgroup::bias_add_op_t<data_type_bias, gpu_arch::Xe>,
          subgroup::
              elemwise_reduce_op_t<reduce_op::sum, data_type_res, gpu_arch::Xe>,
          subgroup::elemwise_reduce_op_t<
              reduce_op::sum,
              data_type_res,
              gpu_arch::Xe>>;
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
          DQUANT_S,
          post_op>;
      constexpr uint32_t barrier_count = hgemm_wint4_functor::barrier_count;
      constexpr uint32_t slm_size = hgemm_wint4_functor::slm_size;

      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      const data_type_b* b_alias = reinterpret_cast<const bit4x2*>(b);
      const data_type_b* b_zp_alias = reinterpret_cast<const bit4x2*>(b_zp);

      hgemm_wint4_functor::run(
          ei,
          const_cast<scalar_t*>(a),
          const_cast<bit4x2*>(b_alias),
          out,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          const_cast<bit4x2*>(b_zp_alias),
          const_cast<scalar_t*>(b_scale),
          DQUANT_S == 0 ? 1 : k / DQUANT_S,
          {{{const_cast<scalar_t*>(bias), {n, 1, n}},
            {const_cast<scalar_t*>(res0), {n, m, n}},
            {const_cast<scalar_t*>(res1), {n, m, n}}}});
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3>
inline void hgemm_qkv_wint4(
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
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = n / 2;
  uint32_t ldc = n;
  cl::sycl::range<3> GroupRange{3, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);
      using data_type_a = scalar_t;
      using data_type_b = bit4x2;
      using data_type_c = scalar_t;
      using data_type_zp = bit4x2;
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
          DQUANT_S,
          post_op>;
      constexpr uint32_t barrier_count = hgemm_wint4_functor::barrier_count;
      constexpr uint32_t slm_size = hgemm_wint4_functor::slm_size;

      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      uint32_t batch_id = ei.get_group(0);
      scalar_t* c = (batch_id == 0) ? out0 : ((batch_id == 1) ? out1 : out2);
      uint32_t weight_offset = batch_id * k * n / 2;

      uint32_t group_num = 1;
      if constexpr (DQUANT_S != 0) {
        group_num = k / DQUANT_S;
      }
      uint32_t zp_offset = batch_id * group_num * n / 2;
      uint32_t scale_offset = batch_id * group_num * n;

      const data_type_b* b_alias = reinterpret_cast<const bit4x2*>(b);
      const data_type_b* b_zp_alias = reinterpret_cast<const bit4x2*>(b_zp);

      hgemm_wint4_functor::run(
          ei,
          const_cast<scalar_t*>(a),
          const_cast<bit4x2*>(b_alias + weight_offset),
          c,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          const_cast<bit4x2*>(b_zp_alias + zp_offset),
          const_cast<scalar_t*>(b_scale + scale_offset),
          group_num);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int DQUANT_S = 1,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3>
inline void hgemm_qkv_bias_wint4(
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
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = n / 2;
  uint32_t ldc = n;
  cl::sycl::range<3> GroupRange{3, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);
      using data_type_a = scalar_t;
      using data_type_b = bit4x2;
      using data_type_c = scalar_t;
      using data_type_zp = bit4x2;
      using data_type_scale = scalar_t;
      using data_type_acc = float;
      using data_type_bias = scalar_t;
      using post_op = subgroup::chained_tile_op_t<
          subgroup::bias_add_op_t<data_type_bias, gpu_arch::Xe>>;
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
          DQUANT_S,
          post_op>;
      constexpr uint32_t barrier_count = hgemm_wint4_functor::barrier_count;
      constexpr uint32_t slm_size = hgemm_wint4_functor::slm_size;

      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      uint32_t batch_id = ei.get_group(0);
      scalar_t* c = (batch_id == 0) ? out0 : ((batch_id == 1) ? out1 : out2);
      uint32_t bias_offset = batch_id * n;
      uint32_t weight_offset = batch_id * k * n / 2;

      uint32_t group_num = 1;
      if constexpr (DQUANT_S != 0) {
        group_num = k / DQUANT_S;
      }
      uint32_t zp_offset = batch_id * group_num * n / 2;
      uint32_t scale_offset = batch_id * group_num * n;

      const data_type_b* b_alias = reinterpret_cast<const bit4x2*>(b);
      const data_type_b* b_zp_alias = reinterpret_cast<const bit4x2*>(b_zp);

      hgemm_wint4_functor::run(
          ei,
          const_cast<scalar_t*>(a),
          const_cast<bit4x2*>(b_alias + weight_offset),
          c,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          const_cast<bit4x2*>(b_zp_alias + zp_offset),
          const_cast<scalar_t*>(b_scale + scale_offset),
          group_num,
          {{{const_cast<scalar_t*>(bias + bias_offset), {n, 1, n}}}});
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace xetla
} // namespace xpu
