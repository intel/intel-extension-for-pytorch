#pragma once

#include <utils/DPCPP.h>
#include "../xetla.h"

namespace xpu {
namespace xetla {

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <typename dtype_in_>
struct alpha_beta_t {
  using dtype_in = dtype_in_;
  using mem_desc_in_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_in_t::shape_t;
  using coord_t = typename mem_desc_in_t::coord_t;
  using base_t = typename mem_desc_in_t::base_t;

  struct arguments_t {
    shape_t shape;
    base_t base;
    dtype_in alpha, beta;
    inline arguments_t() = default;
    inline arguments_t(
        base_t base_,
        shape_t shape_,
        dtype_in alpha_,
        dtype_in beta_)
        : base(base_), shape(shape_), alpha(alpha_), beta(beta_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr int32_t num_block_y = matAcc_t::num_block_y;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using mat_in_tile_desc_t = tile_desc_t<
        tile_size_x,
        tile_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
    using mat_in_payload_t = mem_payload_t<
        dtype_in,
        mat_in_tile_desc_t,
        msg_type_v<mat_in_tile_desc_t, mem_desc_in_t::space>,
        mem_desc_in_t::layout,
        mem_desc_in_t::space,
        gpu_arch::Xe>;
    using mat_in_tile_acc_t = tile_t<dtype_acc, mat_in_tile_desc_t>;
    mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
    mat_in_tile_t mat_in;
    mat_in_payload_t mat_in_payload(mem_desc_in);
    tile_load<cache_hint::cached, cache_hint::cached>(mat_in, mat_in_payload);
    mat_in_tile_acc_t mat_in_acc;
    elemwise_cvt(mat_in_acc, mat_in);

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto src_reg = mat_in_acc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        dst_reg = args.alpha * dst_reg + args.beta * src_reg;
      }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr int32_t tail_size_y = tile_size_y % block_size_y;
      constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto src_reg = mat_in_acc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        dst_reg = args.alpha * dst_reg + args.beta * src_reg;
      }
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
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = false>
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<alpha_beta_t<data_type_c>>,
              update_method,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
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
          {{{const_cast<scalar_t*>(res), {n, m, n}, alpha, beta}}});
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
inline void hgemm_common(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const int m,
    const int n,
    const int k) {
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<>,
              update_method,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<bias_add_op_t<data_type_bias, gpu_arch::Xe>>,
              update_method,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
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
          {{{const_cast<scalar_t*>(bias), {n, 1, n}}}});
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
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
          {{{const_cast<scalar_t*>(bias), {n, 1, n}}, {}}});
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
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
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
          {{{const_cast<scalar_t*>(bias), {n, 1, n}},
            {const_cast<scalar_t*>(res0), {n, m, n}},
            {const_cast<scalar_t*>(res1), {n, m, n}}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename dtype_in_>
struct xres_op_t {
  using dtype_in = dtype_in_;
  using mem_desc_in_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_in_t::shape_t;
  using coord_t = typename mem_desc_in_t::coord_t;
  using base_t = typename mem_desc_in_t::base_t;

  struct arguments_t {
    shape_t shape;
    base_t base;
    dtype_in x;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_, dtype_in x_)
        : base(base_), shape(shape_), x(x_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr int32_t num_block_y = matAcc_t::num_block_y;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using mat_in_tile_desc_t = tile_desc_t<
        tile_size_x,
        tile_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
    using mat_in_payload_t = mem_payload_t<
        dtype_in,
        mat_in_tile_desc_t,
        msg_type_v<mat_in_tile_desc_t, mem_desc_in_t::space>,
        mem_desc_in_t::layout,
        mem_desc_in_t::space,
        gpu_arch::Xe>;
    using mat_in_tile_acc_t = tile_t<dtype_acc, mat_in_tile_desc_t>;
    mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
    mat_in_tile_t mat_in;
    mat_in_payload_t mat_in_payload(mem_desc_in);
    tile_load<cache_hint::cached, cache_hint::cached>(mat_in, mat_in_payload);
    mat_in_tile_acc_t mat_in_acc;
    elemwise_cvt(mat_in_acc, mat_in);

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto src_reg = mat_in_acc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        dst_reg = dst_reg + args.x * src_reg;
      }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr int32_t tail_size_y = tile_size_y % block_size_y;
      constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto src_reg = mat_in_acc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        dst_reg = dst_reg + args.x * src_reg;
      }
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
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = false>
inline void hgemm_bias_res(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const scalar_t* res,
    const scalar_t res_scale,
    const int m,
    const int n,
    const int k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<
                  bias_add_op_t<data_type_bias, gpu_arch::Xe>,
                  xres_op_t<data_type_res>>,
              update_method,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
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
          {{{const_cast<scalar_t*>(bias), {n, 1, n}},
            {const_cast<scalar_t*>(res), {n, m, n}, res_scale}}});
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
  cl::sycl::range<3> GroupRange{3, group_range_m, group_range_n};
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<>,
              update_method,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
      using gemm_op_t = gemm_t<
          dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
          brgemm_t,
          epilogue_t>;

      uint32_t batch_id = ei.get_group(0);
      scalar_t* out = (batch_id == 0) ? out0 : ((batch_id == 1) ? out1 : out2);

      uint32_t size_b = k * n;

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
    const int k) {
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
  cl::sycl::range<3> GroupRange{3, group_range_m, group_range_n};
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<bias_add_op_t<data_type_bias, gpu_arch::Xe>>,
              update_method,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
      using gemm_op_t = gemm_t<
          dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
          brgemm_t,
          epilogue_t>;

      uint32_t batch_id = ei.get_group(0);
      scalar_t* out = (batch_id == 0) ? out0 : ((batch_id == 1) ? out1 : out2);

      uint32_t size_b = k * n;
      uint32_t size_bias = n;

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
          {{{const_cast<scalar_t*>(bias) + size_bias * batch_id, {n, 1, n}}}});
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
inline void hgemm_mul(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* mul,
    const int m,
    const int n,
    const int k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
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
          {{{const_cast<scalar_t*>(mul), {n, m, n}}}});
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
inline void hgemm_silu(
    sycl::queue& queue,
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const int m,
    const int n,
    const int k) {
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<silu_op_t>,
              update_method,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
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
  static_assert(L3_KS == 1, "for fused op, L3_KS should be 1");
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
      using update_method = typename std::
          conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
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
          {{{const_cast<scalar_t*>(res), {n, m, n}}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace xetla
} // namespace xpu
