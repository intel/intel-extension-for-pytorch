#pragma once

#include <utils/DPCPP.h>
#include "../xetla.h"
#include "epilogue_impl.h"

namespace xpu {
namespace xetla {

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR,
    typename tile_op_t>
struct hgemm_caller {
  using data_type_b = scalar_t;
  using data_type_a = scalar_t;
  using data_type_c = scalar_t;
  using data_type_acc = float;
  using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
  using epilogue_t = epilogue_t<
      epilogue_policy_tile_op<tile_op_t, gpu_arch::Xe>,
      tile_shape,
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
  using args_t = epilogue_t::arguments_t;

  void operator()(
      sycl::queue& queue,
      scalar_t* out,
      const scalar_t* a,
      const scalar_t* b,
      const int m,
      const int n,
      const int k,
      args_t args) {
    static_assert(L3_KS == 1, "currently, L3_KS should be 1");
    constexpr mem_layout layout_a = mem_layout::row_major;
    constexpr mem_layout layout_b =
        B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
    uint32_t group_range_m = (m + WG_M - 1) / WG_M;
    uint32_t group_range_n = (n + WG_N - 1) / WG_N;
    uint32_t thread_range_m = WG_M / SG_M;
    uint32_t thread_range_n = WG_N / SG_N;
    uint32_t lda = k;
    uint32_t ldb = B_ROW_MAJOR ? n : k;
    uint32_t ldc = n;
    cl::sycl::range<3> GroupRange{L3_KS, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
    cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

    static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
    static constexpr uint32_t prefetch_distance = STAGES;

    using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
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

    using gemm_op_t = gemm_t<
        dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
        brgemm_t,
        epilogue_t>;

    typename gemm_op_t::arguments_t arg(
        m,
        k,
        n,
        const_cast<scalar_t*>(a),
        lda,
        const_cast<scalar_t*>(b),
        ldb,
        out,
        ldc,
        args);

    auto cgf = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
        xetla_exec_item<3> ei(item);
        slm_barrier_init<gemm_op_t>();
        gemm_op_t gemm_op;
        gemm_op(ei, arg);
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
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_addmm(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* res,
    const scalar_t* a,
    const scalar_t* b,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float beta) {
  using tile_op_t = chained_tile_op_t<epilogue_impl::alpha_beta_op_t<scalar_t>>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue,
      out,
      a,
      b,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(res), {n, m, n}, alpha, beta}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_common(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const int m,
    const int n,
    const int k) {
  using tile_op_t = chained_tile_op_t<>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(queue, out, a, b, m, n, k, {{}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_res(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* res,
    const int m,
    const int n,
    const int k,
    const float res_factor) {
  using tile_op_t = chained_tile_op_t<epilogue_impl::res_op_t<scalar_t>>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue,
      out,
      a,
      b,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(res), {n, m, n}, res_factor}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_res_res(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* res0,
    const scalar_t* res1,
    const int m,
    const int n,
    const int k,
    const float res0_factor,
    const float res1_factor) {
  using tile_op_t = chained_tile_op_t<
      epilogue_impl::res_op_t<scalar_t>,
      epilogue_impl::res_op_t<scalar_t>>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue,
      out,
      a,
      b,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(res0), {n, m, n}, res0_factor},
        {const_cast<scalar_t*>(res1), {n, m, n}, res1_factor}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_bias(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const int m,
    const int n,
    const int k,
    const float bias_factor) {
  using tile_op_t = chained_tile_op_t<epilogue_impl::bias_op_t<scalar_t>>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue,
      out,
      a,
      b,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_bias_res(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const scalar_t* res,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res_factor) {
  using tile_op_t = chained_tile_op_t<
      epilogue_impl::bias_op_t<scalar_t>,
      epilogue_impl::res_op_t<scalar_t>>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue,
      out,
      a,
      b,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor},
        {const_cast<scalar_t*>(res), {n, m, n}, res_factor}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_bias_res_res(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const scalar_t* res0,
    const scalar_t* res1,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res0_factor,
    const float res1_factor) {
  using tile_op_t = chained_tile_op_t<
      epilogue_impl::bias_op_t<scalar_t>,
      epilogue_impl::res_op_t<scalar_t>,
      epilogue_impl::res_op_t<scalar_t>>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue,
      out,
      a,
      b,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor},
        {const_cast<scalar_t*>(res0), {n, m, n}, res0_factor},
        {const_cast<scalar_t*>(res1), {n, m, n}, res1_factor}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_bias_relu(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const int m,
    const int n,
    const int k,
    const float bias_factor) {
  using tile_op_t =
      chained_tile_op_t<epilogue_impl::bias_op_t<scalar_t>, relu_op_t>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue,
      out,
      a,
      b,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor}, {}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_bias_gelu(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const int m,
    const int n,
    const int k,
    const float bias_factor) {
  using tile_op_t =
      chained_tile_op_t<epilogue_impl::bias_op_t<scalar_t>, gelu_fwd_op_t>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue,
      out,
      a,
      b,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor}, {}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_mul(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* mul,
    const int m,
    const int n,
    const int k) {
  using tile_op_t = chained_tile_op_t<
      elemwise_reduce_op_t<reduce_op::prod, scalar_t, gpu_arch::Xe>>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(
      queue, out, a, b, m, n, k, {{{const_cast<scalar_t*>(mul), {n, m, n}}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_silu(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const int m,
    const int n,
    const int k) {
  using tile_op_t = chained_tile_op_t<epilogue_impl::silu_op_t>;
  auto caller = hgemm_caller<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      SYNC_FREQ,
      STAGES,
      B_ROW_MAJOR,
      tile_op_t>();
  caller(queue, out, a, b, m, n, k, {{{}}});
}

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_qkv(
    sycl::queue& queue,
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const scalar_t* b,
    const int m,
    const int n,
    const int k,
    const int group) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  constexpr mem_layout layout_a = mem_layout::row_major;
  constexpr mem_layout layout_b =
      B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = B_ROW_MAJOR ? n : k;
  uint32_t ldc = n;
  uint32_t size_b = k * n;
  uint32_t size_o = m * n;
  cl::sycl::range<3> GroupRange{group, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
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
      using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;

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
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<chained_tile_op_t<>, gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
      using gemm_op_t = gemm_t<
          dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
          brgemm_t,
          epilogue_t>;

      uint32_t batch_id = ei.get_group(0);
      slm_barrier_init<gemm_op_t>();
      scalar_t* out = (batch_id <= group - 3)
          ? out0 + batch_id * size_o
          : ((batch_id == group - 2) ? out1 : out2);

      typename gemm_op_t::arguments_t arg(
          m,
          k,
          n,
          const_cast<scalar_t*>(a),
          lda,
          const_cast<scalar_t*>(b) + size_b * batch_id,
          ldb,
          out,
          ldc);
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
    int SG_K,
    int SLM_KS,
    int L3_KS,
    int SYNC_FREQ,
    int STAGES,
    bool B_ROW_MAJOR>
inline void hgemm_qkv_bias(
    sycl::queue& queue,
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const int m,
    const int n,
    const int k,
    const int group) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  constexpr mem_layout layout_a = mem_layout::row_major;
  constexpr mem_layout layout_b =
      B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = B_ROW_MAJOR ? n : k;
  uint32_t ldc = n;
  uint32_t size_o = m * n;
  uint32_t size_b = k * n;
  uint32_t size_bias = n;
  cl::sycl::range<3> GroupRange{group, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);

      using data_type_b = scalar_t;
      using data_type_a = scalar_t;
      using data_type_c = scalar_t;
      using data_type_bias = scalar_t;
      using data_type_acc = float;
      static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
      static constexpr uint32_t prefetch_distance = STAGES;
      using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;

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
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<epilogue_impl::bias_op_t<data_type_bias>>,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
      using gemm_op_t = gemm_t<
          dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
          brgemm_t,
          epilogue_t>;

      uint32_t batch_id = ei.get_group(0);
      slm_barrier_init<gemm_op_t>();
      scalar_t* out = (batch_id <= group - 3)
          ? out0 + batch_id * size_o
          : ((batch_id == group - 2) ? out1 : out2);

      typename gemm_op_t::arguments_t arg(
          m,
          k,
          n,
          const_cast<scalar_t*>(a),
          lda,
          const_cast<scalar_t*>(b) + size_b * batch_id,
          ldb,
          out,
          ldc,
          {{{const_cast<scalar_t*>(bias) + size_bias * batch_id,
             {n, 1, n},
             {1}}}});
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace xetla
} // namespace xpu
