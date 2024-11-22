#pragma once

#include <CL/sycl.hpp>
#include <torch/torch.h>
#include <xetla.hpp>
#include <vector>
#include "../../moe_gemm.h"

namespace gpu {
namespace xetla {

using cgfs_t = torch_ipex::xpu::xetla::cgfs_t;

template <typename T, int N, int Start>
inline typename std::enable_if_t<(N == Start), xetla_vector<T, N>>
inclusive_prefix_sum(xetla_vector<T, N> src) {
  return src;
}

template <typename T, int N, int Start>
inline typename std::enable_if_t<(Start != N), xetla_vector<T, N>>
inclusive_prefix_sum(xetla_vector<T, N> src) {
  // assert N is a power of 2
  static_assert((N & (N - 1)) == 0, "N is expected to be power of 2");
  xetla_vector<T, N> dst = src;
  dst.xetla_select<N - Start, 1>(Start) += src.xetla_select<N - Start, 1>(0);
  return inclusive_prefix_sum<T, N, Start * 2>(dst);
}

template <typename T, int ExpertNum, typename Policy, gpu_arch arch_tag>
struct MoEGEMM {
  static constexpr int wg_tile_m = Policy::wg_tile_m;
  static constexpr int wg_tile_n = Policy::wg_tile_n;
  static constexpr int sg_tile_m = Policy::sg_tile_m;
  static constexpr int sg_tile_n = Policy::sg_tile_n;
  static constexpr int k_stride = Policy::k_stride;
  static constexpr int stages = Policy::stages;
  static constexpr int sync_freq = Policy::sync_freq;

  static constexpr int num_sub_group_per_wg =
      (wg_tile_m / sg_tile_m) * (wg_tile_n / sg_tile_n);

  using mem_desc_t = mem_desc_t<T, mem_layout::row_major, mem_space::global>;
  using accum_t = float;

  using compute_attr = group::compute_attr_t<T, T, accum_t>;
  using perf_tuning_knob =
      group::perf_tuning_knob_t<k_stride, stages, sync_freq>;
  using compute_policy = group::
      compute_policy_default_xmx<compute_attr, perf_tuning_knob, arch_tag>;
  using tile_shape =
      group::tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

  using gemm_t =
      group::gemm_t<compute_policy, tile_shape, mem_desc_t, mem_desc_t>;
  using gemm_args_t = typename gemm_t::arguments_t;
  using matAcc_t = typename gemm_t::matAcc_t;
  using work_group_t = typename gemm_t::work_group_t;
  using epilogue_t = group::epilogue_t<
      group::epilogue_policy_default<arch_tag>,
      tile_shape,
      mem_desc_t>;
  static constexpr uint32_t barrier_count = gemm_t::barrier_count;
  static constexpr uint32_t slm_size = gemm_t::slm_size;

  MoEGEMM(
      const T* activation,
      const T* weights,
      T* outputs,
      const int gemm_n,
      const int gemm_k,
      const int* total_rows_for_each_expert)
      : activation(activation),
        weights(weights),
        outputs(outputs),
        gemm_n(gemm_n),
        gemm_k(gemm_k),
        total_rows_for_each_expert(total_rows_for_each_expert) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int* total_rows_for_each_expert_h,
      const int gemm_n) {
    int tile_n = (gemm_n + wg_tile_n - 1) / wg_tile_n;
    int total_tile_m = 0;
    for (int i = 0; i < ExpertNum; ++i) {
      int gemm_m = total_rows_for_each_expert_h[i];
      int tile_m = (gemm_m + wg_tile_m - 1) / wg_tile_m;
      total_tile_m += tile_m;
    }

    sycl::range<3> local(1, 1, num_sub_group_per_wg);
    sycl::range<3> global(total_tile_m, tile_n, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  template <int N = ExpertNum>
  inline std::enable_if_t<N == 1, void> get_current_tile_info(
      const xetla_vector<int, N>& rows_for_experts,
      const int group_m_id,
      int* expert_id,
      int* expert_m_id,
      int* skip_m) const {
    return;
  }

  template <int N = ExpertNum>
  inline std::enable_if_t<(N > 1), void> get_current_tile_info(
      const xetla_vector<int, ExpertNum>& rows_for_experts,
      const int group_m_id,
      int* expert_id,
      int* expert_m_id,
      int* skip_m) const {
    xetla_vector<int, ExpertNum> cumsum_rows_for_experts =
        inclusive_prefix_sum<int, ExpertNum, 1>(rows_for_experts);

    xetla_vector<int, ExpertNum> cumsum_tiles_for_experts =
        inclusive_prefix_sum<int, ExpertNum, 1>(
            (rows_for_experts + wg_tile_m - 1) / wg_tile_m);

    xetla_vector<uint32_t, ExpertNum> mask =
        group_m_id >= cumsum_tiles_for_experts;
    uint32_t expert_start =
        sycl::ext::intel::esimd::cbit(sycl::ext::intel::esimd::ballot(mask));

    if (expert_start == 0) {
      *expert_id = 0;
      *expert_m_id = group_m_id;
      *skip_m = 0;
      return;
    }

    *expert_id = expert_start;
    *expert_m_id = group_m_id - cumsum_tiles_for_experts[expert_start - 1];
    *skip_m = cumsum_rows_for_experts[expert_start - 1];
  }

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    int group_m_id = item.get_group(0);
    int group_n_id = item.get_group(1);

    xetla_nbarrier_init<barrier_count>();
    xetla_local_init<slm_size>();

    int expert_id = 0;
    int expert_m_id = group_m_id;
    int skip_m = 0;
    xetla_vector<int, ExpertNum> rows_for_experts =
        xetla_load_global<int, ExpertNum>((int*)total_rows_for_each_expert, 0);
    get_current_tile_info(
        rows_for_experts, group_m_id, &expert_id, &expert_m_id, &skip_m);

    const T* current_weights = weights + expert_id * gemm_n * gemm_k;
    const int gemm_m = rows_for_experts[expert_id];

    mem_desc_t mem_desc_a, mem_desc_b, mem_desc_c;
    int start_x = group_n_id * wg_tile_n;
    int start_y = skip_m + expert_m_id * wg_tile_m;
    mem_desc_a.init(
        (T*)activation,
        {static_cast<uint32_t>(gemm_k),
         static_cast<uint32_t>(skip_m + gemm_m),
         static_cast<uint32_t>(gemm_k)},
        {0, start_y});
    mem_desc_b.init(
        (T*)current_weights,
        {static_cast<uint32_t>(gemm_n),
         static_cast<uint32_t>(gemm_k),
         static_cast<uint32_t>(gemm_n)},
        {start_x, 0});
    mem_desc_c.init(
        (T*)outputs,
        {static_cast<uint32_t>(gemm_n),
         static_cast<uint32_t>(skip_m + gemm_m),
         static_cast<uint32_t>(gemm_n)},
        {start_x, start_y});

    gemm_t gemm;
    uint32_t loop_count = (gemm_k + k_stride - 1) / k_stride;
    gemm_args_t gemm_args(mem_desc_a, mem_desc_b, loop_count);
    matAcc_t matAcc(0);
    work_group_t g(item.get_local_linear_id());
    gemm(g, matAcc, gemm_args);

    epilogue_t epilogue;
    epilogue(g, matAcc, mem_desc_c);
  }

  const T* activation;
  const T* weights;
  T* outputs;
  const int gemm_n;
  const int gemm_k;
  const int* total_rows_for_each_expert;
};

template <
    typename T,
    int ExpertNum,
    typename Policy,
    gpu_arch arch_tag = gpu_arch::XeHpc>
cgfs_t LaunchMoEGEMM(
    sycl::queue& queue,
    const T* activation,
    const T* weights,
    T* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_each_expert,
    const int* total_rows_for_each_expert_h) {
  using kernel = MoEGEMM<T, ExpertNum, Policy, arch_tag>;
  auto cgf = [=](sycl::handler& cgh) {
    kernel task(
        activation,
        weights,
        outputs,
        gemm_n,
        gemm_k,
        total_rows_for_each_expert);
    cgh.parallel_for(
        kernel::get_nd_range(total_rows_for_each_expert_h, gemm_n), task);
  };
  return {cgf};
}

} // namespace xetla
} // namespace gpu