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

template <typename T, typename Policy, gpu_arch arch_tag>
struct MoEGEMM {
  static constexpr int wg_tile_m = Policy::wg_tile_m;
  static constexpr int wg_tile_n = Policy::wg_tile_n;
  static constexpr int sg_tile_m = Policy::sg_tile_m;
  static constexpr int sg_tile_n = Policy::sg_tile_n;
  static constexpr int k_stride = Policy::k_stride;
  static constexpr int stages = Policy::stages;
  static constexpr int sync_freq = Policy::sync_freq;
  static constexpr int load_expert_num = 8;

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
      const int* total_rows_for_each_expert,
      const int expert_num)
      : activation(activation),
        weights(weights),
        outputs(outputs),
        gemm_n(gemm_n),
        gemm_k(gemm_k),
        total_rows_for_each_expert(total_rows_for_each_expert),
        expert_num(expert_num) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int* total_rows_for_each_expert_h,
      const int gemm_n,
      const int expert_num) {
    int tile_n = (gemm_n + wg_tile_n - 1) / wg_tile_n;
    int total_tile_m = 0;
    for (int i = 0; i < expert_num; ++i) {
      int gemm_m = total_rows_for_each_expert_h[i];
      int tile_m = (gemm_m + wg_tile_m - 1) / wg_tile_m;
      total_tile_m += tile_m;
    }

    sycl::range<3> local(1, 1, num_sub_group_per_wg);
    sycl::range<3> global(total_tile_m, tile_n, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    int group_m_id = item.get_group(0);
    int group_n_id = item.get_group(1);

    xetla_nbarrier_init<barrier_count>();
    xetla_local_init<slm_size>();

    int expert_id = 0;
    int expert_m_id = group_m_id;
    int skip_m = 0;

    int pre_rows = 0;
    int pre_tiles = 0;
    int gemm_m = 0;
    for (int i = 0; i < expert_num; i += load_expert_num) {
      xetla_vector<int, load_expert_num> rows_for_experts =
          xetla_load_global<int, load_expert_num>(
              (int*)total_rows_for_each_expert, i * sizeof(int));

      xetla_vector<int, load_expert_num> cumsum_rows_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(rows_for_experts);

      xetla_vector<int, load_expert_num> cumsum_tiles_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(
              (rows_for_experts + wg_tile_m - 1) / wg_tile_m);

      cumsum_rows_for_experts += pre_rows;
      cumsum_tiles_for_experts += pre_tiles;

      if (group_m_id >= cumsum_tiles_for_experts[load_expert_num - 1]) {
        pre_rows = cumsum_rows_for_experts[load_expert_num - 1];
        pre_tiles = cumsum_tiles_for_experts[load_expert_num - 1];
        continue;
      }

      xetla_vector<uint32_t, load_expert_num> mask =
          group_m_id >= cumsum_tiles_for_experts;

      uint32_t load_start =
          sycl::ext::intel::esimd::cbit(sycl::ext::intel::esimd::ballot(mask));

      uint32_t expert_start = load_start + i;

      if (load_start == 0) {
        expert_m_id = group_m_id - pre_tiles;
        skip_m = pre_rows;
      } else {
        expert_m_id = group_m_id - cumsum_tiles_for_experts[load_start - 1];
        skip_m = cumsum_rows_for_experts[load_start - 1];
      }
      expert_id = expert_start;
      gemm_m = rows_for_experts[load_start];
      break;
    }

    int64_t offset = static_cast<int64_t>(expert_id) *
        static_cast<int64_t>(gemm_n) * static_cast<int64_t>(gemm_k);
    const T* current_weights = weights + offset;

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
  const int expert_num;
};

template <typename T, typename Policy, gpu_arch arch_tag = gpu_arch::XeHpc>
cgfs_t LaunchMoEGEMM(
    sycl::queue& queue,
    const T* activation,
    const T* weights,
    T* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_each_expert,
    const int* total_rows_for_each_expert_h,
    const int expert_num) {
  using kernel = MoEGEMM<T, Policy, arch_tag>;
  auto cgf = [=](sycl::handler& cgh) {
    kernel task(
        activation,
        weights,
        outputs,
        gemm_n,
        gemm_k,
        total_rows_for_each_expert,
        expert_num);
    cgh.parallel_for(
        kernel::get_nd_range(total_rows_for_each_expert_h, gemm_n, expert_num),
        task);
  };
  return {cgf};
}

template <
    typename T,
    typename Policy,
    fp8_format f_format,
    bool vnni_t,
    gpu_arch arch_tag>
struct MoEGEMMFP8 {
  static constexpr int wg_tile_m = Policy::wg_tile_m;
  static constexpr int wg_tile_n = Policy::wg_tile_n;
  static constexpr int sg_tile_m = Policy::sg_tile_m;
  static constexpr int sg_tile_n = Policy::sg_tile_n;
  static constexpr int k_stride = Policy::k_stride;
  static constexpr int stages = Policy::stages;
  static constexpr int sync_freq = Policy::sync_freq;
  static constexpr int load_expert_num = 8;

  static constexpr int num_sub_group_per_wg =
      (wg_tile_m / sg_tile_m) * (wg_tile_n / sg_tile_n);

  using mem_desc_A_t = mem_desc_t<T, mem_layout::row_major, mem_space::global>;
  using mem_desc_B_t =
      mem_desc_t<uint8_t, mem_layout::row_major, mem_space::global>;
  using accum_t = float;

  using compute_attr = group::compute_attr_t<T, T, accum_t>;
  using perf_tuning_knob =
      group::perf_tuning_knob_t<k_stride, stages, sync_freq>;
  using compute_policy = group::compute_policy_fp8_dequantize<
      compute_attr,
      perf_tuning_knob,
      f_format,
      vnni_t,
      arch_tag>;
  using tile_shape =
      group::tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

  using gemm_t =
      group::gemm_t<compute_policy, tile_shape, mem_desc_A_t, mem_desc_B_t>;
  using gemm_args_t = typename gemm_t::arguments_t;
  using matAcc_t = typename gemm_t::matAcc_t;
  using work_group_t = typename gemm_t::work_group_t;
  using epilogue_t = group::epilogue_t<
      group::epilogue_policy_default<arch_tag>,
      tile_shape,
      mem_desc_A_t>;
  static constexpr uint32_t barrier_count = gemm_t::barrier_count;
  static constexpr uint32_t slm_size = gemm_t::slm_size;

  MoEGEMMFP8(
      const T* activation,
      const uint8_t* weights,
      const float* scale,
      T* outputs,
      const int gemm_n,
      const int gemm_k,
      const int* total_rows_for_each_expert,
      const int expert_num)
      : activation(activation),
        weights(weights),
        scale(scale),
        outputs(outputs),
        gemm_n(gemm_n),
        gemm_k(gemm_k),
        total_rows_for_each_expert(total_rows_for_each_expert),
        expert_num(expert_num) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int* total_rows_for_each_expert_h,
      const int gemm_n,
      const int expert_num) {
    int tile_n = (gemm_n + wg_tile_n - 1) / wg_tile_n;
    int total_tile_m = 0;
    for (int i = 0; i < expert_num; ++i) {
      int gemm_m = total_rows_for_each_expert_h[i];
      int tile_m = (gemm_m + wg_tile_m - 1) / wg_tile_m;
      total_tile_m += tile_m;
    }

    sycl::range<3> local(1, 1, num_sub_group_per_wg);
    sycl::range<3> global(total_tile_m, tile_n, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    int group_m_id = item.get_group(0);
    int group_n_id = item.get_group(1);

    xetla_nbarrier_init<barrier_count>();
    xetla_local_init<slm_size>();

    int expert_id = 0;
    int expert_m_id = group_m_id;
    int skip_m = 0;

    int pre_rows = 0;
    int pre_tiles = 0;
    int gemm_m = 0;
    for (int i = 0; i < expert_num; i += load_expert_num) {
      xetla_vector<int, load_expert_num> rows_for_experts =
          xetla_load_global<int, load_expert_num>(
              (int*)total_rows_for_each_expert, i * sizeof(int));

      xetla_vector<int, load_expert_num> cumsum_rows_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(rows_for_experts);

      xetla_vector<int, load_expert_num> cumsum_tiles_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(
              (rows_for_experts + wg_tile_m - 1) / wg_tile_m);

      cumsum_rows_for_experts += pre_rows;
      cumsum_tiles_for_experts += pre_tiles;

      if (group_m_id >= cumsum_tiles_for_experts[load_expert_num - 1]) {
        pre_rows = cumsum_rows_for_experts[load_expert_num - 1];
        pre_tiles = cumsum_tiles_for_experts[load_expert_num - 1];
        continue;
      }

      xetla_vector<uint32_t, load_expert_num> mask =
          group_m_id >= cumsum_tiles_for_experts;

      uint32_t load_start =
          sycl::ext::intel::esimd::cbit(sycl::ext::intel::esimd::ballot(mask));

      uint32_t expert_start = load_start + i;

      if (load_start == 0) {
        expert_m_id = group_m_id - pre_tiles;
        skip_m = pre_rows;
      } else {
        expert_m_id = group_m_id - cumsum_tiles_for_experts[load_start - 1];
        skip_m = cumsum_rows_for_experts[load_start - 1];
      }
      expert_id = expert_start;
      gemm_m = rows_for_experts[load_start];
      break;
    }

    int64_t offset = static_cast<int64_t>(expert_id) *
        static_cast<int64_t>(gemm_n) * static_cast<int64_t>(gemm_k);
    const uint8_t* current_weights = weights + offset;
    const float current_scale = scale[expert_id];

    mem_desc_A_t mem_desc_a, mem_desc_c;
    mem_desc_B_t mem_desc_b;
    int start_x = group_n_id * wg_tile_n;
    int start_y = skip_m + expert_m_id * wg_tile_m;
    mem_desc_a.init(
        (T*)activation,
        {static_cast<uint32_t>(gemm_k),
         static_cast<uint32_t>(skip_m + gemm_m),
         static_cast<uint32_t>(gemm_k)},
        {0, start_y});
    mem_desc_b.init(
        (uint8_t*)current_weights,
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
    gemm_args_t gemm_args(mem_desc_a, mem_desc_b, loop_count, current_scale);
    matAcc_t matAcc(0);
    work_group_t g(item.get_local_linear_id());
    gemm(g, matAcc, gemm_args);

    epilogue_t epilogue;
    epilogue(g, matAcc, mem_desc_c);
  }

  const T* activation;
  const uint8_t* weights;
  const float* scale;
  T* outputs;
  const int gemm_n;
  const int gemm_k;
  const int* total_rows_for_each_expert;
  const int expert_num;
};

template <
    typename T,
    fp8_format f_format,
    bool vnni_t,
    typename Policy,
    gpu_arch arch_tag = gpu_arch::XeHpc>
cgfs_t LaunchMoEGEMMFP8(
    sycl::queue& queue,
    const T* activation,
    const uint8_t* weights,
    const float* scale,
    T* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_each_expert,
    const int* total_rows_for_each_expert_h,
    const int expert_num) {
  using kernel = MoEGEMMFP8<T, Policy, f_format, vnni_t, arch_tag>;
  auto cgf = [=](sycl::handler& cgh) {
    kernel task(
        activation,
        weights,
        scale,
        outputs,
        gemm_n,
        gemm_k,
        total_rows_for_each_expert,
        expert_num);
    cgh.parallel_for(
        kernel::get_nd_range(total_rows_for_each_expert_h, gemm_n, expert_num),
        task);
  };
  return {cgf};
}

template <typename T, typename Policy, gpu_arch arch_tag>
struct PesistantMoEGEMM {
  static constexpr int wg_tile_m = Policy::wg_tile_m;
  static constexpr int wg_tile_n = Policy::wg_tile_n;
  static constexpr int sg_tile_m = Policy::sg_tile_m;
  static constexpr int sg_tile_n = Policy::sg_tile_n;
  static constexpr int k_stride = Policy::k_stride;
  static constexpr int stages = Policy::stages;
  static constexpr int sync_freq = Policy::sync_freq;
  static constexpr int load_expert_num = 8;
  static constexpr int PERSISTENT_SG_NUMS = 20 * 32;
  static constexpr int atomic_slm_size = 1 * sizeof(int);

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
  static constexpr uint32_t gemm_slm_size = gemm_t::slm_size;
  static constexpr uint32_t slm_size = gemm_slm_size + atomic_slm_size;

  PesistantMoEGEMM(
      const T* activation,
      const T* weights,
      T* outputs,
      const int gemm_n,
      const int gemm_k,
      const int* total_rows_for_each_expert,
      int* atomic_buffer,
      const int expert_num)
      : activation(activation),
        weights(weights),
        outputs(outputs),
        gemm_n(gemm_n),
        gemm_k(gemm_k),
        total_rows_for_each_expert(total_rows_for_each_expert),
        atomic_buffer(atomic_buffer),
        expert_num(expert_num) {}

  static inline sycl::nd_range<3> get_nd_range() {
    int group_size = PERSISTENT_SG_NUMS / num_sub_group_per_wg;
    assert(num_sub_group_per_wg * group_size == PERSISTENT_SG_NUMS);

    sycl::range<3> local(1, 1, num_sub_group_per_wg);
    sycl::range<3> global(1, group_size, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    int group_id = item.get_group_linear_id();
    int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
    int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    int group_range = item.get_group_range(1);

    int local_id = item.get_local_linear_id();

    if (group_id == 0 && local_id == 0) {
      __ESIMD_NS::simd<int, 1> value = 0;
      __ESIMD_NS::simd<uint32_t, 1> offset = 0;
      __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::store>(
          atomic_buffer, offset, value);
    }

    xetla_nbarrier_init<barrier_count>();
    xetla_local_init<slm_size>();

    int expert_id = 0;
    int expert_m_id = group_m_id;
    int skip_m = 0;

    int pre_rows = 0;
    int pre_tiles = 0;
    int gemm_m = 0;
    for (int i = 0; i < expert_num; i += load_expert_num) {
      xetla_vector<int, load_expert_num> rows_for_experts =
          xetla_load_global<int, load_expert_num>(
              (int*)total_rows_for_each_expert, i * sizeof(int));

      xetla_vector<int, load_expert_num> cumsum_rows_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(rows_for_experts);

      xetla_vector<int, load_expert_num> cumsum_tiles_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(
              (rows_for_experts + wg_tile_m - 1) / wg_tile_m);

      cumsum_rows_for_experts += pre_rows;
      cumsum_tiles_for_experts += pre_tiles;

      if (group_m_id >= cumsum_tiles_for_experts[load_expert_num - 1]) {
        pre_rows = cumsum_rows_for_experts[load_expert_num - 1];
        pre_tiles = cumsum_tiles_for_experts[load_expert_num - 1];
        continue;
      }

      while (group_m_id < cumsum_tiles_for_experts[load_expert_num - 1]) {
        xetla_vector<uint32_t, load_expert_num> mask =
            group_m_id >= cumsum_tiles_for_experts;

        uint32_t load_start = sycl::ext::intel::esimd::cbit(
            sycl::ext::intel::esimd::ballot(mask));

        uint32_t expert_start = load_start + i;

        if (load_start == 0) {
          expert_m_id = group_m_id - pre_tiles;
          skip_m = pre_rows;
        } else {
          expert_m_id = group_m_id - cumsum_tiles_for_experts[load_start - 1];
          skip_m = cumsum_rows_for_experts[load_start - 1];
        }
        expert_id = expert_start;
        gemm_m = rows_for_experts[load_start];

        int64_t offset = static_cast<int64_t>(expert_id) *
            static_cast<int64_t>(gemm_n) * static_cast<int64_t>(gemm_k);
        const T* current_weights = weights + offset;

        mem_desc_t mem_desc_a, mem_desc_b, mem_desc_c;
        int start_x = (group_id * wg_tile_n) % gemm_n_pad;
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
        if (local_id == 0) {
          __ESIMD_NS::simd<int, 1> value = 1;
          __ESIMD_NS::simd<uint32_t, 1> offset = 0;
          int old_val = __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::add>(
              atomic_buffer, offset, value)[0];
          __ESIMD_NS::slm_block_store<int, 1>(gemm_slm_size, old_val);
        }
        item.barrier(sycl::access::fence_space::local_space);
        group_id = __ESIMD_NS::slm_block_load<int, 1>(gemm_slm_size)[0];
        group_id += group_range;
        group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
      }
      pre_rows = cumsum_rows_for_experts[load_expert_num - 1];
      pre_tiles = cumsum_tiles_for_experts[load_expert_num - 1];
    }
  }

  const T* activation;
  const T* weights;
  T* outputs;
  const int gemm_n;
  const int gemm_k;
  const int* total_rows_for_each_expert;
  int* atomic_buffer;
  const int expert_num;
};

template <typename T, typename Policy, gpu_arch arch_tag = gpu_arch::XeHpc>
cgfs_t LaunchPesistantMoEGEMM(
    sycl::queue& queue,
    const T* activation,
    const T* weights,
    T* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_each_expert,
    int* atomic_buffer,
    const int expert_num) {
  using kernel = PesistantMoEGEMM<T, Policy, arch_tag>;
  auto cgf = [=](sycl::handler& cgh) {
    kernel task(
        activation,
        weights,
        outputs,
        gemm_n,
        gemm_k,
        total_rows_for_each_expert,
        atomic_buffer,
        expert_num);
    cgh.parallel_for(kernel::get_nd_range(), task);
  };
  return {cgf};
}

template <typename T, typename Policy, fp8_format f_format, gpu_arch arch_tag>
struct PesistantMoEGEMMFP8 {
  static constexpr int wg_tile_m = Policy::wg_tile_m;
  static constexpr int wg_tile_n = Policy::wg_tile_n;
  static constexpr int sg_tile_m = Policy::sg_tile_m;
  static constexpr int sg_tile_n = Policy::sg_tile_n;
  static constexpr int k_stride = Policy::k_stride;
  static constexpr int stages = Policy::stages;
  static constexpr int sync_freq = Policy::sync_freq;
  static constexpr int load_expert_num = 8;
  static constexpr int PERSISTENT_SG_NUMS = 20 * 32;
  static constexpr int atomic_slm_size = 1 * sizeof(int);

  static constexpr int num_sub_group_per_wg =
      (wg_tile_m / sg_tile_m) * (wg_tile_n / sg_tile_n);

  using mem_desc_A_t = mem_desc_t<T, mem_layout::row_major, mem_space::global>;
  using mem_desc_B_t =
      mem_desc_t<uint8_t, mem_layout::row_major, mem_space::global>;
  using accum_t = float;

  using compute_attr = group::compute_attr_t<T, T, accum_t>;
  using perf_tuning_knob =
      group::perf_tuning_knob_t<k_stride, stages, sync_freq>;
  using compute_policy = group::compute_policy_fp8_dequantize<
      compute_attr,
      perf_tuning_knob,
      f_format,
      false,
      arch_tag>;
  using tile_shape =
      group::tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

  using gemm_t =
      group::gemm_t<compute_policy, tile_shape, mem_desc_A_t, mem_desc_B_t>;
  using gemm_args_t = typename gemm_t::arguments_t;
  using matAcc_t = typename gemm_t::matAcc_t;
  using work_group_t = typename gemm_t::work_group_t;
  using epilogue_t = group::epilogue_t<
      group::epilogue_policy_default<arch_tag>,
      tile_shape,
      mem_desc_A_t>;
  static constexpr uint32_t barrier_count = gemm_t::barrier_count;
  static constexpr uint32_t gemm_slm_size = gemm_t::slm_size;
  static constexpr uint32_t slm_size = gemm_slm_size + atomic_slm_size;

  PesistantMoEGEMMFP8(
      const T* activation,
      const uint8_t* weights,
      const float* scale,
      T* outputs,
      const int gemm_n,
      const int gemm_k,
      const int* total_rows_for_each_expert,
      int* atomic_buffer,
      const int expert_num)
      : activation(activation),
        weights(weights),
        scale(scale),
        outputs(outputs),
        gemm_n(gemm_n),
        gemm_k(gemm_k),
        total_rows_for_each_expert(total_rows_for_each_expert),
        atomic_buffer(atomic_buffer),
        expert_num(expert_num) {}

  static inline sycl::nd_range<3> get_nd_range() {
    int group_size = PERSISTENT_SG_NUMS / num_sub_group_per_wg;
    assert(num_sub_group_per_wg * group_size == PERSISTENT_SG_NUMS);

    sycl::range<3> local(1, 1, num_sub_group_per_wg);
    sycl::range<3> global(1, group_size, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    int group_id = item.get_group_linear_id();
    int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
    int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    int group_range = item.get_group_range(1);

    int local_id = item.get_local_linear_id();

    if (group_id == 0 && local_id == 0) {
      __ESIMD_NS::simd<int, 1> value = 0;
      __ESIMD_NS::simd<uint32_t, 1> offset = 0;
      __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::store>(
          atomic_buffer, offset, value);
    }

    xetla_nbarrier_init<barrier_count>();
    xetla_local_init<slm_size>();

    int expert_id = 0;
    int expert_m_id = group_m_id;
    int skip_m = 0;

    int pre_rows = 0;
    int pre_tiles = 0;
    int gemm_m = 0;
    for (int i = 0; i < expert_num; i += load_expert_num) {
      xetla_vector<int, load_expert_num> rows_for_experts =
          xetla_load_global<int, load_expert_num>(
              (int*)total_rows_for_each_expert, i * sizeof(int));

      xetla_vector<int, load_expert_num> cumsum_rows_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(rows_for_experts);

      xetla_vector<int, load_expert_num> cumsum_tiles_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(
              (rows_for_experts + wg_tile_m - 1) / wg_tile_m);

      cumsum_rows_for_experts += pre_rows;
      cumsum_tiles_for_experts += pre_tiles;

      if (group_m_id >= cumsum_tiles_for_experts[load_expert_num - 1]) {
        pre_rows = cumsum_rows_for_experts[load_expert_num - 1];
        pre_tiles = cumsum_tiles_for_experts[load_expert_num - 1];
        continue;
      }

      while (group_m_id < cumsum_tiles_for_experts[load_expert_num - 1]) {
        xetla_vector<uint32_t, load_expert_num> mask =
            group_m_id >= cumsum_tiles_for_experts;

        uint32_t load_start = sycl::ext::intel::esimd::cbit(
            sycl::ext::intel::esimd::ballot(mask));

        uint32_t expert_start = load_start + i;

        if (load_start == 0) {
          expert_m_id = group_m_id - pre_tiles;
          skip_m = pre_rows;
        } else {
          expert_m_id = group_m_id - cumsum_tiles_for_experts[load_start - 1];
          skip_m = cumsum_rows_for_experts[load_start - 1];
        }
        expert_id = expert_start;
        gemm_m = rows_for_experts[load_start];

        int64_t offset = static_cast<int64_t>(expert_id) *
            static_cast<int64_t>(gemm_n) * static_cast<int64_t>(gemm_k);
        const uint8_t* current_weights = weights + offset;
        const float current_scale = scale[expert_id];

        mem_desc_A_t mem_desc_a, mem_desc_c;
        mem_desc_B_t mem_desc_b;

        int start_x = (group_id * wg_tile_n) % gemm_n_pad;
        int start_y = skip_m + expert_m_id * wg_tile_m;
        mem_desc_a.init(
            (T*)activation,
            {static_cast<uint32_t>(gemm_k),
             static_cast<uint32_t>(skip_m + gemm_m),
             static_cast<uint32_t>(gemm_k)},
            {0, start_y});
        mem_desc_b.init(
            (uint8_t*)current_weights,
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
        gemm_args_t gemm_args(
            mem_desc_a, mem_desc_b, loop_count, current_scale);
        matAcc_t matAcc(0);
        work_group_t g(item.get_local_linear_id());
        gemm(g, matAcc, gemm_args);

        epilogue_t epilogue;
        epilogue(g, matAcc, mem_desc_c);
        if (local_id == 0) {
          __ESIMD_NS::simd<int, 1> value = 1;
          __ESIMD_NS::simd<uint32_t, 1> offset = 0;
          int old_val = __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::add>(
              atomic_buffer, offset, value)[0];
          __ESIMD_NS::slm_block_store<int, 1>(gemm_slm_size, old_val);
        }
        item.barrier(sycl::access::fence_space::local_space);
        group_id = __ESIMD_NS::slm_block_load<int, 1>(gemm_slm_size)[0];
        group_id += group_range;
        group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
      }
      pre_rows = cumsum_rows_for_experts[load_expert_num - 1];
      pre_tiles = cumsum_tiles_for_experts[load_expert_num - 1];
    }
  }

  const T* activation;
  const uint8_t* weights;
  const float* scale;
  T* outputs;
  const int gemm_n;
  const int gemm_k;
  const int* total_rows_for_each_expert;
  int* atomic_buffer;
  const int expert_num;
};

template <
    typename T,
    fp8_format f_format,
    typename Policy,
    gpu_arch arch_tag = gpu_arch::XeHpc>
cgfs_t LaunchPesistantMoEGEMMFP8(
    sycl::queue& queue,
    const T* activation,
    const uint8_t* weights,
    const float* scale,
    T* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_each_expert,
    int* atomic_buffer,
    const int expert_num) {
  using kernel = PesistantMoEGEMMFP8<T, Policy, f_format, arch_tag>;
  auto cgf = [=](sycl::handler& cgh) {
    kernel task(
        activation,
        weights,
        scale,
        outputs,
        gemm_n,
        gemm_k,
        total_rows_for_each_expert,
        atomic_buffer,
        expert_num);
    cgh.parallel_for(kernel::get_nd_range(), task);
  };
  return {cgf};
}

} // namespace xetla
} // namespace gpu