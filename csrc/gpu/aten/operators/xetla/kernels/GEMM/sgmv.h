#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <xetla.hpp>

namespace gpu {
namespace xetla {

// Maybe need to add split-K for better performance since
// n is small while k is large in LoRA shrink scenario.
struct SgmvShrinkPolicy {
  static constexpr uint32_t wg_tile_m = 128;
  static constexpr uint32_t wg_tile_n = 32;
  static constexpr uint32_t sg_tile_m = 32;
  static constexpr uint32_t sg_tile_n = 16;
  static constexpr uint32_t k_stride = 16;
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t sync_freq = 0;
};

struct SgmvExpandPolicy {
  static constexpr uint32_t wg_tile_m = 256;
  static constexpr uint32_t wg_tile_n = 256;
  static constexpr uint32_t sg_tile_m = 32;
  static constexpr uint32_t sg_tile_n = 32;
  static constexpr uint32_t k_stride = 16;
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t sync_freq = 0;
};

template <
    typename output_t,
    typename input_t,
    typename Policy,
    gpu_arch arch_tag>
struct SgmvShrinkKernel {
  static constexpr uint32_t wg_tile_m = Policy::wg_tile_m;
  static constexpr uint32_t wg_tile_n = Policy::wg_tile_n;
  static constexpr uint32_t sg_tile_m = Policy::sg_tile_m;
  static constexpr uint32_t sg_tile_n = Policy::sg_tile_n;
  static constexpr uint32_t k_stride = Policy::k_stride;
  static constexpr uint32_t stages = Policy::stages;
  static constexpr uint32_t sync_freq = Policy::sync_freq;

  static constexpr uint32_t num_sub_group_per_wg =
      (wg_tile_m / sg_tile_m) * (wg_tile_n / sg_tile_n);

  using input_mem_desc_t =
      mem_desc_t<input_t, mem_layout::row_major, mem_space::global>;
  using weight_mem_desc_t =
      mem_desc_t<input_t, mem_layout::col_major, mem_space::global>;
  using output_mem_desc_t =
      mem_desc_t<output_t, mem_layout::row_major, mem_space::global>;

  using accum_t = float;
  using compute_attr = group::compute_attr_t<input_t, input_t, accum_t>;
  using perf_tuning_knob =
      group::perf_tuning_knob_t<k_stride, stages, sync_freq>;
  using compute_policy = group::
      compute_policy_default_xmx<compute_attr, perf_tuning_knob, arch_tag>;
  using tile_shape =
      group::tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

  using gemm_t = group::
      gemm_t<compute_policy, tile_shape, input_mem_desc_t, weight_mem_desc_t>;
  using gemm_args_t = typename gemm_t::arguments_t;
  using matAcc_t = typename gemm_t::matAcc_t;
  using work_group_t = typename gemm_t::work_group_t;
  using epilogue_t = group::epilogue_t<
      group::epilogue_policy_default<arch_tag>,
      tile_shape,
      output_mem_desc_t>;
  static constexpr uint32_t barrier_count = gemm_t::barrier_count;
  static constexpr uint32_t slm_size = gemm_t::slm_size;

  inline sycl::nd_range<3> get_nd_range() {
    tile_m = (max_seq_len + wg_tile_m - 1) / wg_tile_m;
    tile_n = (gemm_n + wg_tile_n - 1) / wg_tile_n;

    sycl::range<3> local(1, 1, num_sub_group_per_wg);
    sycl::range<3> global(batches, tile_m * tile_n, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    uint32_t batch_id = item.get_group(0);
    uint32_t mn_linear_id = item.get_group(1);
    uint32_t group_m_id = mn_linear_id / tile_n;
    uint32_t group_n_id = mn_linear_id % tile_n;

    uint32_t seq_len = seq_lens[batch_id];
    if (group_m_id * wg_tile_m > seq_len)
      return;

    uint32_t lora_id = lora_indices[batch_id];
    if (lora_id < 0)
      return;

    xetla_nbarrier_init<barrier_count>();
    xetla_local_init<slm_size>();

    uint32_t seq_start_loc = seq_start_locs[batch_id];
    input_t* current_weights = weights + lora_id * gemm_n * gemm_k;

    input_mem_desc_t mem_desc_a;
    weight_mem_desc_t mem_desc_b;
    output_mem_desc_t mem_desc_c;
    int start_x = group_n_id * wg_tile_n;
    int start_y = seq_start_loc + group_m_id * wg_tile_m;
    mem_desc_a.init(
        inputs,
        {static_cast<uint32_t>(gemm_k),
         static_cast<uint32_t>(seq_start_loc + seq_len),
         static_cast<uint32_t>(gemm_k)},
        {0, start_y});
    mem_desc_b.init(
        current_weights,
        {static_cast<uint32_t>(gemm_n),
         static_cast<uint32_t>(gemm_k),
         static_cast<uint32_t>(gemm_k)},
        {start_x, 0});
    mem_desc_c.init(
        outputs,
        {static_cast<uint32_t>(gemm_n),
         static_cast<uint32_t>(seq_start_loc + seq_len),
         static_cast<uint32_t>(gemm_n)},
        {start_x, start_y});

    gemm_t gemm;
    uint32_t loop_count = (gemm_k + k_stride - 1) / k_stride;
    gemm_args_t gemm_args(mem_desc_a, mem_desc_b, loop_count);
    matAcc_t matAcc(0);
    work_group_t g(item.get_local_linear_id());
    gemm(g, matAcc, gemm_args);
    matAcc.reg *= scale;

    epilogue_t epilogue;
    epilogue(g, matAcc, mem_desc_c);
  }

  SgmvShrinkKernel(
      output_t* outputs,
      input_t* inputs,
      input_t* weights,
      int64_t* seq_start_locs,
      int64_t* seq_lens,
      int64_t* lora_indices,
      uint32_t batches,
      uint32_t max_seq_len,
      uint32_t gemm_k,
      uint32_t gemm_n,
      float scale)
      : outputs(outputs),
        inputs(inputs),
        weights(weights),
        seq_start_locs(seq_start_locs),
        seq_lens(seq_lens),
        lora_indices(lora_indices),
        batches(batches),
        max_seq_len(max_seq_len),
        gemm_k(gemm_k),
        gemm_n(gemm_n),
        scale(scale) {}

 private:
  output_t* outputs;
  input_t* inputs;
  input_t* weights;
  int64_t* seq_start_locs;
  int64_t* seq_lens;
  int64_t* lora_indices;
  uint32_t batches;
  uint32_t max_seq_len;
  uint32_t gemm_k;
  uint32_t gemm_n;
  float scale;
  uint32_t tile_m;
  uint32_t tile_n;
};

template <
    typename output_t,
    typename input_t,
    typename Policy,
    gpu_arch arch_tag>
struct SgmvExpandKernel {
  static constexpr uint32_t wg_tile_m = Policy::wg_tile_m;
  static constexpr uint32_t wg_tile_n = Policy::wg_tile_n;
  static constexpr uint32_t sg_tile_m = Policy::sg_tile_m;
  static constexpr uint32_t sg_tile_n = Policy::sg_tile_n;
  static constexpr uint32_t k_stride = Policy::k_stride;
  static constexpr uint32_t stages = Policy::stages;
  static constexpr uint32_t sync_freq = Policy::sync_freq;

  static constexpr uint32_t num_sub_group_per_wg =
      (wg_tile_m / sg_tile_m) * (wg_tile_n / sg_tile_n);

  using input_mem_desc_t =
      mem_desc_t<input_t, mem_layout::row_major, mem_space::global>;
  using weight_mem_desc_t =
      mem_desc_t<output_t, mem_layout::col_major, mem_space::global>;
  using output_mem_desc_t =
      mem_desc_t<output_t, mem_layout::row_major, mem_space::global>;

  using accum_t = float;
  using compute_attr = group::compute_attr_t<output_t, output_t, accum_t>;
  using perf_tuning_knob =
      group::perf_tuning_knob_t<k_stride, stages, sync_freq>;
  using compute_policy = group::
      compute_policy_default_xmx<compute_attr, perf_tuning_knob, arch_tag>;
  using tile_shape =
      group::tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

  using gemm_t = group::
      gemm_t<compute_policy, tile_shape, input_mem_desc_t, weight_mem_desc_t>;
  using gemm_args_t = typename gemm_t::arguments_t;
  using matAcc_t = typename gemm_t::matAcc_t;
  using work_group_t = typename gemm_t::work_group_t;
  using epilogue_t = group::epilogue_t<
      group::epilogue_policy_default<arch_tag>,
      tile_shape,
      output_mem_desc_t>;
  static constexpr uint32_t barrier_count = gemm_t::barrier_count;
  static constexpr uint32_t slm_size = gemm_t::slm_size;

  inline sycl::nd_range<3> get_nd_range() {
    tile_m = (max_seq_len + wg_tile_m - 1) / wg_tile_m;
    tile_n = (gemm_n + wg_tile_n - 1) / wg_tile_n;

    sycl::range<3> local(1, 1, num_sub_group_per_wg);
    sycl::range<3> global(batches, tile_m * tile_n, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  void operator()(sycl::nd_item<3> item) const SYCL_ESIMD_KERNEL {
    uint32_t batch_id = item.get_group(0);
    uint32_t mn_linear_id = item.get_group(1);
    uint32_t group_m_id = mn_linear_id / tile_n;
    uint32_t group_n_id = mn_linear_id % tile_n;

    uint32_t seq_len = seq_lens[batch_id];
    if (group_m_id * wg_tile_m > seq_len)
      return;

    uint32_t lora_id = lora_indices[batch_id];
    if (lora_id < 0)
      return;

    xetla_nbarrier_init<barrier_count>();
    xetla_local_init<slm_size>();

    uint32_t seq_start_loc = seq_start_locs[batch_id];
    output_t* current_weights = weights + lora_id * gemm_n * gemm_k;

    input_mem_desc_t mem_desc_a;
    weight_mem_desc_t mem_desc_b;
    output_mem_desc_t mem_desc_c;
    int start_x = group_n_id * wg_tile_n;
    int start_y = seq_start_loc + group_m_id * wg_tile_m;
    mem_desc_a.init(
        inputs,
        {static_cast<uint32_t>(gemm_k),
         static_cast<uint32_t>(seq_start_loc + seq_len),
         static_cast<uint32_t>(gemm_k)},
        {0, start_y});
    mem_desc_b.init(
        current_weights,
        {static_cast<uint32_t>(gemm_n),
         static_cast<uint32_t>(gemm_k),
         static_cast<uint32_t>(gemm_k)},
        {start_x, 0});
    mem_desc_c.init(
        outputs,
        {static_cast<uint32_t>(output_hidden),
         static_cast<uint32_t>(seq_start_loc + seq_len),
         static_cast<uint32_t>(output_hidden)},
        {slice_offset + start_x, start_y});

    gemm_t gemm;
    uint32_t loop_count = (gemm_k + k_stride - 1) / k_stride;
    gemm_args_t gemm_args(mem_desc_a, mem_desc_b, loop_count);
    matAcc_t matAcc(0);
    work_group_t g(item.get_local_linear_id());
    gemm(g, matAcc, gemm_args);

    if (add_to_output) {
      using mat_tile_desc = typename matAcc_t::tile_desc;
      using matC_t = subgroup::tile_t<output_t, mat_tile_desc>;
      using matC_payload_t = subgroup::mem_payload_t<
          output_mem_desc_t,
          mat_tile_desc,
          msg_type::block_2d,
          arch_tag>;

      int32_t sg_idx = g.get_id() % tile_shape::wg_size_x;
      int32_t sg_idy = g.get_id() / tile_shape::wg_size_x;
      int32_t tile_offset_n = sg_idx * tile_shape::sg_tile_size_x;
      int32_t tile_offset_m = sg_idy * tile_shape::sg_tile_size_y;
      mem_desc_c.update_coord(tile_offset_n, tile_offset_m);

      matC_t matC;
      matC_payload_t matC_payload(mem_desc_c);
      subgroup::tile_load(matC, matC_payload);

      matC_t mat_to_add;
      subgroup::elemwise_cvt(mat_to_add, matAcc);

      matC.reg += mat_to_add.reg;
      subgroup::tile_store(matC, matC_payload);
    } else {
      epilogue_t epilogue;
      epilogue(g, matAcc, mem_desc_c);
    }
  }

  SgmvExpandKernel(
      output_t* outputs,
      input_t* inputs,
      output_t* weights,
      int64_t* seq_start_locs,
      int64_t* seq_lens,
      int64_t* lora_indices,
      uint32_t batches,
      uint32_t max_seq_len,
      uint32_t gemm_k,
      uint32_t gemm_n,
      uint32_t slice_offset,
      uint32_t output_hidden,
      bool add_to_output)
      : outputs(outputs),
        inputs(inputs),
        weights(weights),
        seq_start_locs(seq_start_locs),
        seq_lens(seq_lens),
        lora_indices(lora_indices),
        batches(batches),
        max_seq_len(max_seq_len),
        gemm_k(gemm_k),
        gemm_n(gemm_n),
        slice_offset(slice_offset),
        output_hidden(output_hidden),
        add_to_output(add_to_output) {}

 private:
  output_t* outputs;
  input_t* inputs;
  output_t* weights;
  int64_t* seq_start_locs;
  int64_t* seq_lens;
  int64_t* lora_indices;
  uint32_t batches;
  uint32_t max_seq_len;
  uint32_t gemm_k;
  uint32_t gemm_n;
  uint32_t slice_offset;
  uint32_t output_hidden;
  bool add_to_output;
  uint32_t tile_m;
  uint32_t tile_n;
};

} // namespace xetla
} // namespace gpu