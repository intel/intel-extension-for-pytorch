#pragma once

#include <utils/DPCPP.h>
#include "../xetla.h"

namespace xpu {
namespace xetla {

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int SG_K = 64,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = false>
inline void hgemm_common(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const int m,
    const int n,
    const int k) {
  uint32_t matrix_m = m;
  uint32_t matrix_n = n;
  uint32_t matrix_k = k;
  constexpr uint32_t slm_kslicing = SLM_KS;
  constexpr uint32_t l3_kslicing = L3_KS;
  constexpr uint32_t wg_tile_m = WG_M;
  constexpr uint32_t wg_tile_n = WG_N;
  constexpr uint32_t sg_tile_m = SG_M;
  constexpr uint32_t sg_tile_n = SG_N;
  constexpr mem_layout layout_a = mem_layout::row_major;
  constexpr mem_layout layout_b =
      B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
  uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
  uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
  uint32_t thread_range_m = wg_tile_m / sg_tile_m;
  uint32_t thread_range_n = wg_tile_n / sg_tile_n;
  uint32_t lda = matrix_k;
  uint32_t ldb = B_ROW_MAJOR ? matrix_n : matrix_k;
  uint32_t ldc = matrix_n;
  cl::sycl::range<3> GroupRange{l3_kslicing, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{slm_kslicing, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);

      using data_type_b = scalar_t;
      using data_type_a = scalar_t;
      using data_type_c = scalar_t;
      using data_type_acc = float;
      static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
      static constexpr uint32_t prefetch_distance = STAGES;
      using tile_shape =
          tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

      using brgemm_t = typename brgemm_selector_t<
          data_type_a,
          data_type_b,
          layout_a,
          layout_b,
          mem_space::global,
          mem_space::global,
          8,
          8,
          data_type_acc,
          tile_shape,
          SG_K,
          mma_engine::xmx,
          gpu_arch::Xe,
          prefetch_distance,
          periodic_sync_interval>::brgemm;
      using update_method = typename std::conditional<
          (l3_kslicing > 1),
          result_reduce_sum,
          result_overwrite>::type;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<>,
              update_method,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
      using gemm_op_t = gemm_t<
          dispatch_policy_kslicing<l3_kslicing, slm_kslicing, gpu_arch::Xe>,
          brgemm_t,
          epilogue_t>;
      typename gemm_op_t::arguments_t arg(
          matrix_m,
          matrix_k,
          matrix_n,
          const_cast<scalar_t*>(a),
          lda,
          const_cast<scalar_t*>(b),
          ldb,
          out,
          ldc);
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int KS,
    int KN,
    bool B_ROW_MAJOR = false>
inline void hgemm_splitk(
    sycl::queue& queue,
    float* acc,
    const scalar_t* a,
    const scalar_t* b,
    const int m,
    const int n,
    const int k) {
  uint32_t matrix_m = m;
  uint32_t matrix_n = n;
  uint32_t matrix_k = k;
  constexpr uint32_t split_k_S = KS;
  constexpr uint32_t wg_tile_m = WG_M;
  constexpr uint32_t wg_tile_n = WG_N;
  constexpr uint32_t sg_tile_m = SG_M;
  constexpr uint32_t sg_tile_n = SG_N;
  uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
  uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
  uint32_t thread_range_m = wg_tile_m / sg_tile_m;
  uint32_t thread_range_n = wg_tile_n / sg_tile_n;
  uint32_t lda = matrix_k;
  uint32_t ldb = B_ROW_MAJOR ? matrix_n : matrix_k;
  uint32_t ldc = matrix_n;
  cl::sycl::range<3> GroupRange{split_k_S, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{1, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);

      using data_type_b = scalar_t;
      using data_type_a = scalar_t;
      using data_type_c = float;
      using data_type_acc = float;
      using compute_attr =
          compute_attr_t<data_type_a, data_type_b, data_type_acc>;

      static constexpr uint32_t periodic_sync_interval = 8;
      static constexpr uint32_t prefetch_distance = 3;
      static constexpr uint32_t k_iter_num = KN;
      using perf_tuning_knob = perf_tuning_knob_t<
          k_iter_num,
          prefetch_distance,
          periodic_sync_interval>;
      using compute_policy = compute_policy_default_xmx<
          compute_attr,
          perf_tuning_knob,
          gpu_arch::Xe>;

      using mem_desc_input_a =
          mem_desc_t<data_type_a, mem_layout::row_major, mem_space::global>;
      using mem_desc_input_b = std::conditional<
          B_ROW_MAJOR,
          mem_desc_t<data_type_b, mem_layout::row_major, mem_space::global>,
          mem_desc_t<data_type_b, mem_layout::col_major, mem_space::global>>::
          type;
      using mem_desc_output_c =
          mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;

      using tile_shape =
          tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
      using brgemm_t = brgemm_t<
          compute_policy,
          tile_shape,
          mem_desc_input_a,
          mem_desc_input_b>;
      brgemm_t brgemm;

      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<>,
              result_reduce_sum,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_output_c>;

      static constexpr uint32_t barrier_count = brgemm_t::barrier_count;
      static constexpr uint32_t slm_size = brgemm_t::slm_size;
      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();
      int start_n = ei.get_group(2) * wg_tile_n;
      int start_m = ei.get_group(1) * wg_tile_m;
      int split_k_R = matrix_k / split_k_S;
      int split_k_E = ei.get_group(0) * split_k_R;
      uint32_t split_k_D = split_k_E + split_k_R;
      int start_k = split_k_E;
      uint32_t wg_tile_k = split_k_R;
      uint32_t inner_loop_count = wg_tile_k / k_iter_num;

      mem_desc_input_a md_a(
          {const_cast<scalar_t*>(a)},
          {split_k_D, matrix_m, lda},
          {start_k, start_m});
      mem_desc_input_b md_b(
          {const_cast<scalar_t*>(b)},
          {matrix_n, split_k_D, ldb},
          {start_n, start_k});
      mem_desc_output_c md_c(
          {acc}, {matrix_n, matrix_m, ldc}, {start_n, start_m});

      typename brgemm_t::matAcc_t matAcc;
      matAcc.init(0);
      typename brgemm_t::arguments_t brgemm_args(md_a, md_b, inner_loop_count);
      typename brgemm_t::work_group_t g(ei.get_local_linear_id());
      brgemm(g, matAcc, brgemm_args);
      epilogue_t epilogue;
      epilogue(g, matAcc, md_c);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace xetla
} // namespace xpu
