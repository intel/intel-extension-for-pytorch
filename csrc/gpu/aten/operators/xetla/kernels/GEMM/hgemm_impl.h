#pragma once

#include "../xetla.h"
#include "epilogue_impl.h"

namespace torch_ipex::xpu::xetla {

template <gpu_arch arch_tag>
class gemm_perf_knob_t {
 public:
  static constexpr uint32_t periodic_sync_interval = 0;
  static constexpr uint32_t prefetch_distance = 0;
};

template <>
class gemm_perf_knob_t<gpu_arch::XeHpc> {
 public:
  static constexpr uint32_t periodic_sync_interval = 8;
  static constexpr uint32_t prefetch_distance = 3;
};

template <
    int WG_M_,
    int WG_N_,
    int SG_M_,
    int SG_N_,
    int SG_K_,
    int SLM_KS_,
    int L3_KS_,
    bool B_ROW_MAJOR_>
struct gemm_tile_policy_t {
  static constexpr uint32_t WG_M = WG_M_;
  static constexpr uint32_t WG_N = WG_N_;
  static constexpr uint32_t SG_M = SG_M_;
  static constexpr uint32_t SG_N = SG_N_;
  static constexpr uint32_t SG_K = SG_K_;
  static constexpr uint32_t SLM_KS = SLM_KS_;
  static constexpr uint32_t L3_KS = L3_KS_;
  static constexpr bool B_ROW_MAJOR = B_ROW_MAJOR_;
};

template <
    typename scalar_t,
    typename gemm_tile_policy,
    typename tile_op_t,
    gpu_arch arch_tag>
struct hgemm_caller {
  using data_type_b = scalar_t;
  using data_type_a = scalar_t;
  using data_type_c = scalar_t;
  using data_type_acc = float;
  static constexpr uint32_t WG_M = gemm_tile_policy::WG_M;
  static constexpr uint32_t WG_N = gemm_tile_policy::WG_N;
  static constexpr uint32_t SG_M = gemm_tile_policy::SG_M;
  static constexpr uint32_t SG_N = gemm_tile_policy::SG_N;
  static constexpr uint32_t SG_K = gemm_tile_policy::SG_K;
  using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;

  using epilogue_t = epilogue_t<
      epilogue_policy_tile_op<tile_op_t, arch_tag>,
      tile_shape,
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
  using args_t = epilogue_t::arguments_t;
  using gemm_perf_knob = gemm_perf_knob_t<arch_tag>;
  static constexpr uint32_t periodic_sync_interval =
      gemm_perf_knob::periodic_sync_interval;
  static constexpr uint32_t prefetch_distance =
      gemm_perf_knob::prefetch_distance;
  static constexpr bool use_xmx =
      arch_has_xmx<arch_tag> && (SG_N >= dpas_attr_t<arch_tag>::n_in_elem);

  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = gemm_tile_policy::B_ROW_MAJOR
      ? mem_layout::row_major
      : mem_layout::col_major;

  using gemm_t = typename gemm_selector_t<
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
      use_xmx ? mma_engine::xmx : mma_engine::fpu,
      arch_tag,
      prefetch_distance,
      periodic_sync_interval>::gemm;
  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<arch_tag>;

  using dispatch_policy = dispatch_policy_kslicing<
      group_swizzle,
      gemm_tile_policy::L3_KS,
      gemm_tile_policy::SLM_KS>;
  using gemm_op_t = gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;

  struct HgemmCallerKernelFunctor {
    KERNEL_MAIN void operator()(nd_item<3> item) const {
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(item, arg);
    }
    HgemmCallerKernelFunctor(gemm_op_t::arguments_t arg_) : arg(arg_) {}

   private:
    gemm_op_t::arguments_t arg;
  };

  cgfs_t operator()(
      scalar_t* out,
      const scalar_t* a,
      const scalar_t* b,
      data_type_acc* acc_ptr,
      uint32_t* cnt_ptr,
      const int m,
      const int n,
      const int k,
      args_t args) {
    uint32_t lda = k;
    uint32_t ldb = gemm_tile_policy::B_ROW_MAJOR ? n : k;
    uint32_t ldc = n;
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
        acc_ptr,
        cnt_ptr,
        args);
    HgemmCallerKernelFunctor kfn(arg);
    sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(m, n);
    return {[=](sycl::handler& cgh) {
      cgh.parallel_for<decltype(kfn)>(NDRange, kfn);
    }};
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_addmm(
    scalar_t* out,
    const scalar_t* res,
    const scalar_t* a,
    const scalar_t* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float beta) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t =
      chained_tile_op_t<epilogue_impl::alpha_beta_op_t<scalar_t, arch_tag>>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_common(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(out, a, b, acc_ptr, cnt_ptr, m, n, k, {{}});
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_res(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float res_factor) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t =
      chained_tile_op_t<epilogue_impl::res_op_t<scalar_t, arch_tag>>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_res_res(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* res0,
    const scalar_t* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float res0_factor,
    const float res1_factor) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<
      epilogue_impl::res_op_t<scalar_t, arch_tag>,
      epilogue_impl::res_op_t<scalar_t, arch_tag>>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_bias(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t =
      chained_tile_op_t<epilogue_impl::bias_op_t<scalar_t, arch_tag>>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_bias_res(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const scalar_t* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res_factor) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<
      epilogue_impl::bias_op_t<scalar_t, arch_tag>,
      epilogue_impl::res_op_t<scalar_t, arch_tag>>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_bias_res_res(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    const scalar_t* res0,
    const scalar_t* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res0_factor,
    const float res1_factor) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<
      epilogue_impl::bias_op_t<scalar_t, arch_tag>,
      epilogue_impl::res_op_t<scalar_t, arch_tag>,
      epilogue_impl::res_op_t<scalar_t, arch_tag>>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_bias_relu(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<
      epilogue_impl::bias_op_t<scalar_t, arch_tag>,
      relu_op_t>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_bias_gelu(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<
      epilogue_impl::bias_op_t<scalar_t, arch_tag>,
      gelu_fwd_op_t>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_mul(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<
      elemwise_reduce_op_t<reduce_op::prod, scalar_t, arch_tag>>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(
      out,
      a,
      b,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      {{{const_cast<scalar_t*>(mul), {n, m, n}}}});
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_silu(
    scalar_t* out,
    const scalar_t* a,
    const scalar_t* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k) {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<subgroup::silu_op_t>;
  using hgemm_caller_t =
      hgemm_caller<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static hgemm_caller_t caller;
  return caller(out, a, b, acc_ptr, cnt_ptr, m, n, k, {{{}}});
}

template <
    typename scalar_t,
    typename gemm_tile_policy,
    typename tile_op_t,
    gpu_arch arch_tag>
struct HgemmQKVGemm {
  static constexpr uint32_t WG_M = gemm_tile_policy::WG_M;
  static constexpr uint32_t WG_N = gemm_tile_policy::WG_N;
  static constexpr uint32_t SG_M = gemm_tile_policy::SG_M;
  static constexpr uint32_t SG_N = gemm_tile_policy::SG_N;
  static constexpr uint32_t SG_K = gemm_tile_policy::SG_K;
  using data_type_b = scalar_t;
  using data_type_a = scalar_t;
  using data_type_c = scalar_t;
  using data_type_acc = float;
  using gemm_perf_knob = gemm_perf_knob_t<arch_tag>;
  static constexpr uint32_t periodic_sync_interval =
      gemm_perf_knob::periodic_sync_interval;
  static constexpr uint32_t prefetch_distance =
      gemm_perf_knob::prefetch_distance;
  static constexpr bool use_xmx =
      arch_has_xmx<arch_tag> && (SG_N >= dpas_attr_t<arch_tag>::n_in_elem);
  using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = gemm_tile_policy::B_ROW_MAJOR
      ? mem_layout::row_major
      : mem_layout::col_major;

  using gemm_t = typename gemm_selector_t<
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
      use_xmx ? mma_engine::xmx : mma_engine::fpu,
      arch_tag,
      prefetch_distance,
      periodic_sync_interval>::gemm;
  using epilogue_t = epilogue_t<
      epilogue_policy_tile_op<tile_op_t, arch_tag>,
      tile_shape,
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<arch_tag>;
  using dispatch_policy = dispatch_policy_kslicing<
      group_swizzle,
      gemm_tile_policy::L3_KS,
      gemm_tile_policy::SLM_KS>;
  using gemm_op_t = gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;

  static constexpr uint32_t thread_range_m = WG_M / SG_M;
  static constexpr uint32_t thread_range_n = WG_N / SG_N;
  static inline cl::sycl::nd_range<3> get_nd_range(
      const uint32_t m,
      const uint32_t n,
      const uint32_t group) {
    static const cl::sycl::range<3> LocalRange{
        gemm_tile_policy::SLM_KS, thread_range_m, thread_range_n};
    const uint32_t group_range_m = (m + WG_M - 1) / WG_M;
    const uint32_t group_range_n = (n + WG_N - 1) / WG_N;
    const cl::sycl::range<3> GroupRange{group, group_range_m, group_range_n};
    return cl::sycl::nd_range<3>{GroupRange * LocalRange, LocalRange};
  };
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
struct HgemmQKVKernelFunctor {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<>;
  using qkv_gemm_t = HgemmQKVGemm<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  using gemm_op_t = qkv_gemm_t::gemm_op_t;
  static inline cl::sycl::nd_range<3> get_nd_range(
      const uint32_t m,
      const uint32_t n,
      const uint32_t group) {
    return qkv_gemm_t::get_nd_range(m, n, group);
  }

  KERNEL_MAIN void operator()(nd_item<3> item) const {
    uint32_t batch_id = item.get_group(0);
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
        acc_ptr,
        cnt_ptr);
    gemm_op_t gemm_op;
    gemm_op(item, arg);
  }

  HgemmQKVKernelFunctor(
      scalar_t* out0,
      scalar_t* out1,
      scalar_t* out2,
      const scalar_t* a,
      const scalar_t* b,
      float* acc_ptr,
      uint32_t* cnt_ptr,
      const int m,
      const int n,
      const int k,
      const int group,
      uint32_t lda,
      uint32_t ldb,
      uint32_t ldc,
      uint32_t size_b,
      uint32_t size_o)
      : out0(out0),
        out1(out1),
        out2(out2),
        a(a),
        b(b),
        acc_ptr(acc_ptr),
        cnt_ptr(cnt_ptr),
        m(m),
        n(n),
        k(k),
        group(group),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        size_b(size_b),
        size_o(size_o) {}

 private:
  scalar_t* out0;
  scalar_t* out1;
  scalar_t* out2;
  const scalar_t* a;
  const scalar_t* b;
  float* acc_ptr;
  uint32_t* cnt_ptr;
  const int m;
  const int n;
  const int k;
  const int group;
  uint32_t lda;
  uint32_t ldb;
  uint32_t ldc;
  uint32_t size_b;
  uint32_t size_o;
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_qkv(
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const scalar_t* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  constexpr uint32_t group = 3;
  uint32_t lda = k;
  uint32_t ldb = B_ROW_MAJOR ? n : k;
  uint32_t ldc = n;
  uint32_t size_b = k * n;
  uint32_t size_o = m * n;
  using hgemm_qkt_t = HgemmQKVKernelFunctor<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR,
      arch_tag>;

  hgemm_qkt_t kfn(
      out0,
      out1,
      out2,
      a,
      b,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      group,
      lda,
      ldb,
      ldc,
      size_b,
      size_o);
  cl::sycl::nd_range<3> NDRange = hgemm_qkt_t::get_nd_range(m, n, group);
  return {
      [=](sycl::handler& cgh) { cgh.parallel_for<hgemm_qkt_t>(NDRange, kfn); }};
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
struct HgemmQKVBiasKernelFunctor {
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t =
      chained_tile_op_t<epilogue_impl::bias_op_t<scalar_t, arch_tag>>;
  using qkv_gemm_t = HgemmQKVGemm<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  static inline cl::sycl::nd_range<3> get_nd_range(
      const uint32_t m,
      const uint32_t n,
      const uint32_t group) {
    return qkv_gemm_t::get_nd_range(m, n, group);
  }

  KERNEL_MAIN void operator()(nd_item<3> item) const {
    using gemm_op_t = qkv_gemm_t::gemm_op_t;
    uint32_t batch_id = item.get_group(0);
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
        acc_ptr,
        cnt_ptr,
        {{{const_cast<scalar_t*>(bias) + size_bias * batch_id,
           {n, 1, n},
           {1}}}});
    gemm_op_t gemm_op;
    gemm_op(item, arg);
  }

  HgemmQKVBiasKernelFunctor(
      scalar_t* out0,
      scalar_t* out1,
      scalar_t* out2,
      const scalar_t* a,
      const scalar_t* b,
      const scalar_t* bias,
      float* acc_ptr,
      uint32_t* cnt_ptr,
      const int m,
      const int n,
      const int k,
      const int group,
      uint32_t lda,
      uint32_t ldb,
      uint32_t ldc,
      uint32_t size_b,
      uint32_t size_o,
      uint32_t size_bias)
      : out0(out0),
        out1(out1),
        out2(out2),
        a(a),
        b(b),
        bias(bias),
        acc_ptr(acc_ptr),
        cnt_ptr(cnt_ptr),
        m(m),
        n(n),
        k(k),
        group(group),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        size_b(size_b),
        size_o(size_o),
        size_bias(size_bias) {}

 private:
  scalar_t* out0;
  scalar_t* out1;
  scalar_t* out2;
  const scalar_t* a;
  const scalar_t* b;
  const scalar_t* bias;
  float* acc_ptr;
  uint32_t* cnt_ptr;
  const int m;
  const int n;
  const int k;
  const int group;
  uint32_t lda;
  uint32_t ldb;
  uint32_t ldc;
  uint32_t size_b;
  uint32_t size_o;
  uint32_t size_bias;
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_qkv_bias(
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  constexpr uint32_t group = 3;
  uint32_t lda = k;
  uint32_t ldb = B_ROW_MAJOR ? n : k;
  uint32_t ldc = n;
  uint32_t size_o = m * n;
  uint32_t size_b = k * n;
  uint32_t size_bias = n;

  using gemm_qkv_bias_t = HgemmQKVBiasKernelFunctor<
      scalar_t,
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR,
      arch_tag>;

  gemm_qkv_bias_t kfn(
      out0,
      out1,
      out2,
      a,
      b,
      bias,
      acc_ptr,
      cnt_ptr,
      m,
      n,
      k,
      group,
      lda,
      ldb,
      ldc,
      size_b,
      size_o,
      size_bias);
  cl::sycl::nd_range<3> NDRange = gemm_qkv_bias_t::get_nd_range(m, n, group);
  return {[=](sycl::handler& cgh) {
    cgh.parallel_for<gemm_qkv_bias_t>(NDRange, kfn);
  }};
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_qkv_group(
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const scalar_t* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const int num_kv_head,
    const int group,
    const int head_dim) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");

  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t = chained_tile_op_t<>;
  using qkv_gemm_t = HgemmQKVGemm<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  using gemm_op_t = qkv_gemm_t::gemm_op_t;

  cl::sycl::nd_range<3> NDRange =
      qkv_gemm_t::get_nd_range(m, head_dim, num_kv_head * group);
  uint32_t lda = k;
  uint32_t ldb = B_ROW_MAJOR ? num_kv_head * group * head_dim : k;
  uint32_t size_b = head_dim;
  uint32_t size_o = head_dim;
  return {[=](sycl::handler& cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) KERNEL_MAIN {
      uint32_t batch_id = item.get_group(0);
      slm_barrier_init<gemm_op_t>();
      scalar_t* out;
      uint32_t ldc;
      auto group_idx = (batch_id / group);
      auto group_off = (batch_id % group);
      if (group_off < group - 2) {
        out = out0 + (group_idx * (group - 2) + group_off) * size_o;
        ldc = num_kv_head * (group - 2) * head_dim;
      } else if (group_off == group - 2) {
        out = out1 + group_idx * size_o;
        ldc = num_kv_head * head_dim;
      } else {
        out = out2 + group_idx * size_o;
        ldc = num_kv_head * head_dim;
      }

      typename gemm_op_t::arguments_t arg(
          m,
          k,
          head_dim,
          const_cast<scalar_t*>(a),
          lda,
          const_cast<scalar_t*>(b) + size_b * batch_id,
          ldb,
          out,
          ldc,
          acc_ptr,
          cnt_ptr);
      gemm_op_t gemm_op;
      gemm_op(item, arg);
    });
  }};
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
    bool B_ROW_MAJOR,
    gpu_arch arch_tag>
inline cgfs_t hgemm_qkv_group_bias(
    scalar_t* out0,
    scalar_t* out1,
    scalar_t* out2,
    const scalar_t* a,
    const scalar_t* b,
    const scalar_t* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const int num_kv_head,
    const int group,
    const int head_dim) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  using tile_policy_t = gemm_tile_policy_t<
      WG_M,
      WG_N,
      SG_M,
      SG_N,
      SG_K,
      SLM_KS,
      L3_KS,
      B_ROW_MAJOR>;
  using tile_op_t =
      chained_tile_op_t<epilogue_impl::bias_op_t<scalar_t, arch_tag>>;
  using qkv_gemm_t = HgemmQKVGemm<scalar_t, tile_policy_t, tile_op_t, arch_tag>;
  using gemm_op_t = qkv_gemm_t::gemm_op_t;

  cl::sycl::nd_range<3> NDRange =
      qkv_gemm_t::get_nd_range(m, head_dim, num_kv_head * group);
  uint32_t lda = k;
  uint32_t ldb = B_ROW_MAJOR ? num_kv_head * group * head_dim : k;
  uint32_t size_b = head_dim;
  uint32_t size_o = head_dim;
  uint32_t size_bias = head_dim;
  return {[=](sycl::handler& cgh) {
    cgh.parallel_for(NDRange, [=](nd_item<3> item) KERNEL_MAIN {
      uint32_t batch_id = item.get_group(0);
      slm_barrier_init<gemm_op_t>();
      scalar_t* out;
      uint32_t ldc;
      auto group_idx = (batch_id / group);
      auto group_off = (batch_id % group);
      if (group_off < (group - 2)) {
        out = out0 + (group_idx * (group - 2) + group_off) * size_o;
        ldc = num_kv_head * (group - 2) * head_dim;
      } else if (group_off == group - 2) {
        out = out1 + group_idx * size_o;
        ldc = num_kv_head * head_dim;
      } else {
        out = out2 + group_idx * size_o;
        ldc = num_kv_head * head_dim;
      }

      typename gemm_op_t::arguments_t arg(
          m,
          k,
          head_dim,
          const_cast<scalar_t*>(a),
          lda,
          const_cast<scalar_t*>(b) + size_b * batch_id,
          ldb,
          out,
          ldc,
          acc_ptr,
          cnt_ptr,
          {{{const_cast<scalar_t*>(bias) + size_bias * batch_id,
             {head_dim, 1, head_dim},
             {1}}}});
      gemm_op_t gemm_op;
      gemm_op(item, arg);
    });
  }};
}

} // namespace torch_ipex::xpu::xetla
