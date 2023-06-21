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
inline void hgemm_bias(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
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
      using data_type_bias = scalar_t;
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
              chained_tile_op_t<bias_add_op_t<data_type_bias, gpu_arch::Xe>>,
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
          ldc,
          {{{const_cast<scalar_t*>(bias), {matrix_n, 1, matrix_n}}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
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
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = false>
inline void hgemm_bias_gelu(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
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
      using data_type_bias = scalar_t;
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
              chained_tile_op_t<
                  bias_add_op_t<data_type_bias, gpu_arch::Xe>,
                  gelu_fwd_op_t>,
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
          ldc,
          {{{const_cast<scalar_t*>(bias), {matrix_n, 1, matrix_n}}, {}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
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
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = false>
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
  static_assert(l3_kslicing == 1, "for fused op, l3_kslicing should be 1");
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
      using data_type_bias = scalar_t;
      using data_type_res = scalar_t;
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
              chained_tile_op_t<
                  bias_add_op_t<data_type_bias, gpu_arch::Xe>,
                  elemwise_reduce_op_t<
                      reduce_op::sum,
                      data_type_res,
                      gpu_arch::Xe>,
                  elemwise_reduce_op_t<
                      reduce_op::sum,
                      data_type_res,
                      gpu_arch::Xe>>,
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
          ldc,
          {{{const_cast<scalar_t*>(bias), {matrix_n, 1, matrix_n}},
            {const_cast<scalar_t*>(res0), {matrix_n, matrix_m, matrix_n}},
            {const_cast<scalar_t*>(res1), {matrix_n, matrix_m, matrix_n}}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
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
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = false>
inline void hgemm_qkv(
    sycl::queue& queue,
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
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
  static_assert(l3_kslicing == 1, "for qkv fusion, l3_kslicing should be 1");
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
  cl::sycl::range<3> GroupRange{3, group_range_m, group_range_n};
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

      uint32_t batch_id = ei.get_group(0);
      scalar_t* out = (batch_id == 0) ? out0 : ((batch_id == 1) ? out1 : out2);

      uint32_t size_b = matrix_k * matrix_n;

      typename gemm_op_t::arguments_t arg(
          matrix_m,
          matrix_k,
          matrix_n,
          const_cast<scalar_t*>(a),
          lda,
          const_cast<scalar_t*>(b) + size_b * batch_id,
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

struct silu_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    using dtype = typename matAcc_t::dtype;
    matAcc.reg = matAcc.reg / (1.f + xetla_exp<dtype>(-1.f * matAcc.reg));
  }
};

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
inline void hgemm_mul(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* mul,
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
  static_assert(l3_kslicing == 1, "for fused op, l3_kslicing should be 1");
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
      using data_type_mul = scalar_t;
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
              chained_tile_op_t<elemwise_reduce_op_t<
                  reduce_op::prod,
                  data_type_mul,
                  gpu_arch::Xe>>,
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
          ldc,
          {{{const_cast<scalar_t*>(mul), {matrix_n, matrix_m, matrix_n}}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
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
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = false>
inline void hgemm_silu(
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
  static_assert(l3_kslicing == 1, "for fused op, l3_kslicing should be 1");
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
              chained_tile_op_t<silu_op_t>,
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
          ldc,
          {{{}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
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
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = false>
inline void hgemm_res(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* res,
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
  static_assert(l3_kslicing == 1, "for fused op, l3_kslicing should be 1");
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
      using data_type_res = scalar_t;
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
              chained_tile_op_t<elemwise_reduce_op_t<
                  reduce_op::sum,
                  data_type_res,
                  gpu_arch::Xe>>,
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
          ldc,
          {{{const_cast<scalar_t*>(res), {matrix_n, matrix_m, matrix_n}}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace xetla
} // namespace xpu
