#pragma once

#include <utils/DPCPP.h>
#include "../xetla.h"
#include "kernel_attr.h"

namespace xpu {
namespace xetla {

template <typename T, typename Act_T>
struct fwd_config_t {
  int input_size;
  int hidden_size;
  int batch_size;
  int group_offset;
  T* layer_ptr =
      nullptr; /// layer_input = sequence_size x batch_size x input_size
  T* hidden_ptr = nullptr; /// workspace for backwards = sequence_size x
                           /// batch_size x hidden_size
  T* hx_ptr = nullptr; /// h_x input = batch_size x hidden_size
  T* layer_output =
      nullptr; /// layer_output = layer_size x batch_size x hidden_size
  T* W_ir_ptr = nullptr;
  T* W_hr_ptr = nullptr;
  T* W_iz_ptr = nullptr;
  T* W_hz_ptr = nullptr;
  T* W_in_ptr = nullptr;
  T* W_hn_ptr = nullptr;
  Act_T* B_ir_ptr = nullptr;
  Act_T* B_iz_ptr = nullptr;
  Act_T* B_hr_ptr = nullptr;
  Act_T* B_hz_ptr = nullptr;
  Act_T* B_in_ptr = nullptr;
  Act_T* B_hn_ptr = nullptr;
  Act_T* mask_ptr = nullptr;
  T* reset_gate_ptr = nullptr;
  T* input_gate_ptr = nullptr;
  T* new_gate_ptr = nullptr;
  T* hgate_2_ptr = nullptr;
  T* cell_out_ptr =
      nullptr; /// cell output = sequence_size x batch_size x hidden_size
  T* h0 = nullptr;
  bool is_final_layer = false;
  bool is_final_cell = false;
  bool is_first_cell = false;
};
/// weights 704 x 3 , 384
#define PART_BRGEMM_CALL(op_id, acc0_ptr, m, k, n, ptr_a, ptr_b1)       \
  wg_tile_k = k;                                                        \
  boundary_k = wg_tile_k;                                               \
  boundary_m = (start_m + wg_tile_m) > m ? m : (start_m + wg_tile_m);   \
  boundary_n = (start_n + wg_tile_n) > n ? n : (start_n + wg_tile_n);   \
  pitch_a = is_col_major_a ? m : k;                                     \
  start_x_a = is_col_major_a ? start_m : start_k;                       \
  start_y_a = is_col_major_a ? start_k : start_m;                       \
                                                                        \
  pitch_b = is_col_major_b ? k : n;                                     \
  start_x_b = is_col_major_b ? start_k : start_n;                       \
  start_y_b = is_col_major_b ? start_n : start_k;                       \
  brgemm_arg.inner_loop_count =                                         \
      (wg_tile_k + sg_tile_k_##op_id - 1) / sg_tile_k_##op_id;          \
  brgemm_arg.matA_base_desc.init(                                       \
      ptr_a, boundary_k, boundary_m, pitch_a, start_x_a, start_y_a);    \
  brgemm_arg.matB_base_desc.init(                                       \
      ptr_b1, boundary_n, boundary_k, pitch_b, start_x_b, start_y_b);   \
                                                                        \
  brgemm_op::call(g, &(brgemm_arg), acc0_ptr /*, acc1_ptr, acc2_ptr*/); \
  SW_BARRIER();

#define MATC_STORE(acc_id, prt_c)                                  \
  matC_base_desc.init(                                             \
      prt_c,                                                       \
      boundary_n,                                                  \
      boundary_m,                                                  \
      hidden_size,                                                 \
      start_n + ei.get_local_id(0) * sg_tile_n,                    \
      start_m + ei.get_local_id(1) * sg_tile_m);                   \
  matC.init_tdesc(matC_base_desc.get_tdesc());                     \
  tile_op::elemwise_cvt<matC_t, matAcc_t>(&matC, &matAcc##acc_id); \
  matC.store();

#define MATH_STORE(acc_id)                                         \
  xetpp_fill_tdesc<T>(                                             \
      matH_base_desc.xetpp_format<uint32_t>(),                     \
      write_base_ptr,                                              \
      boundary_n,                                                  \
      wg_tile_m,                                                   \
      local_pitch,                                                 \
      start_n + brgemm_op::get_matC_offset_x(g),                   \
      brgemm_op::get_matC_offset_y(g));                            \
  matH.init_tdesc(matH_base_desc);                                 \
  tile_op::elemwise_cvt<matH_t, matAcc_t>(&matH, &matAcc##acc_id); \
  matH.store();

using namespace __XETPP_NS;
using namespace __XETPP_BRGEMM_NS;
using namespace __XETPP_GEMM_NS;

template <
    typename T,
    typename Act_T,
    uint32_t wg_tile_m,
    uint32_t wg_tile_n,
    uint32_t sg_tile_m,
    uint32_t sg_tile_n,
    uint32_t sg_tile_k_0,
    uint32_t sg_tile_k_1,
    mem_layout layout_input = mem_layout::row_major,
    mem_layout layout_hidden = mem_layout::row_major,
    mem_layout layout_weight = mem_layout::col_major,
    mem_layout layout_out = mem_layout::row_major,
    mem_space mem_loc_input = mem_space::global,
    mem_space mem_loc_hidden = mem_space::global,
    mem_space mem_loc_weight = mem_space::global,
    mem_space mem_loc_workspace = mem_space::global,
    mem_space mem_loc_out = mem_space::global,
    uint32_t periodic_sync_in_iters = 0>
struct gru_cell {
  static constexpr bool is_from_local_hidden =
      mem_loc_hidden == mem_space::local;
  static constexpr bool is_col_major_a = layout_input == mem_layout::col_major;
  static constexpr bool is_col_major_b = layout_weight == mem_layout::col_major;
  static constexpr uint32_t prefetch_distance = 0;
  static constexpr uint32_t l3_kslicing = 0;
  static constexpr embedded_kind embedded_op_kind =
      layout_input == mem_layout::col_major
      ? embedded_kind::cooperative_slm_tran_a
      : embedded_kind::none;
  using tile_attr = brgemm_tile_attr_t<
      wg_tile_m,
      wg_tile_n,
      sg_tile_m,
      sg_tile_n,
      sg_tile_k_0>;
  using perf_tuning_knob =
      brgemm_perf_tuning_knob_t<periodic_sync_in_iters, prefetch_distance>;
  using gemm_attr =
      gemm_attr_t<tile_attr, perf_tuning_knob, l3_kslicing, embedded_op_kind>;
  using mem_update_config_input = __XETPP_TILE_NS::mem_update_config_t<
      is_col_major_a ? __XETPP_TILE_NS::tdesc_update_dir::y_dir
                     : __XETPP_TILE_NS::tdesc_update_dir::x_dir>;
  using mem_update_config_weight = __XETPP_TILE_NS::mem_update_config_t<
      is_col_major_b ? __XETPP_TILE_NS::tdesc_update_dir::x_dir
                     : __XETPP_TILE_NS::tdesc_update_dir::y_dir>;

  using mem_desc_t =
      brgemm_mem_desc_t<T, mem_layout::row_major, mem_space::global>;

  using mask_desc_t =
      brgemm_mem_desc_t<Act_T, mem_layout::row_major, mem_space::global>;

  using brgemm_mem_attr = brgemm_mem_attr_t<
      layout_input,
      layout_weight,
      mem_loc_input,
      mem_loc_weight,
      offset_mode::const_offset,
      offset_mode::const_offset,
      1,
      1>;

  using embedded_op_attr = embedded_op_attr_t<
      T,
      T,
      tile_attr,
      layout_input,
      layout_weight,
      mem_loc_input,
      mem_loc_weight,
      perf_tuning_knob::prefetch_distance>;
  static constexpr uint32_t tg_size_x = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
  static constexpr uint32_t tg_size_y = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
  using worktile_t = xetpp_worktile_t<tg_size_x * tg_size_y>;
  using arch_attr = arch_attr_t<gpu_arch::Xe>;

  using brgemm_op = xetpp_brgemm_t<
      worktile_t,
      T,
      T,
      T,
      T,
      Act_T,
      tile_attr,
      brgemm_mem_attr,
      accum_op::MMAOp,
      gpu_arch::Xe,
      perf_tuning_knob>;
  using tile_op = xetpp_tile_op_t<gpu_arch::Xe>;
  using brgemm_arguments_input = typename brgemm_op::arguments_t;

  using matAcc_t = typename brgemm_op::matAcc_t;
  using matC_t = __XETPP_TILE_NS::xetpp_tile_store_t<
      T,
      matAcc_t::reg_tile_size_x,
      matAcc_t::reg_tile_size_y,
      matAcc_t::block_size_x,
      matAcc_t::block_size_y,
      layout_out,
      mem_loc_workspace,
      __XETPP_TILE_NS::store_op::normal,
      gpu_arch::Xe,
      __XETPP_TILE_NS::reg_layout::tiled>;

  using matH_t = __XETPP_TILE_NS::xetpp_tile_store_t<
      T,
      matAcc_t::reg_tile_size_x,
      matAcc_t::reg_tile_size_y,
      matAcc_t::block_size_x,
      matAcc_t::block_size_y,
      layout_out,
      mem_loc_out,
      __XETPP_TILE_NS::store_op::normal,
      gpu_arch::Xe,
      __XETPP_TILE_NS::reg_layout::tiled>;

  using mat_t = __XETPP_TILE_NS::xetpp_tile_load_t<
      T,
      matAcc_t::reg_tile_size_x,
      matAcc_t::reg_tile_size_y,
      matAcc_t::block_size_x,
      matAcc_t::block_size_y,
      layout_hidden,
      mem_loc_hidden,
      gpu_arch::Xe,
      __XETPP_TILE_NS::reg_layout::tiled>;

  using mask_t = xetpp_tile_load_t<
      Act_T,
      matAcc_t::reg_tile_size_x,
      matAcc_t::reg_tile_size_y,
      matAcc_t::block_size_x,
      matAcc_t::block_size_y,
      mem_layout::row_major,
      mem_space::global,
      gpu_arch::Xe,
      __XETPP_TILE_NS::reg_layout::tiled>;

  static void inline call(xetpp_exec_item<3> ei, fwd_config_t<T, Act_T>* args) {
    matH_t matH;
    matC_t matC;

    mem_desc_t matC_base_desc;
    mem_desc_t matH_base_desc;
    matAcc_t matAcc0;
    matAcc_t matAcc1;
    matAcc_t matAcc2;
    int batch_size, input_size, hidden_size;
    batch_size = args->batch_size;
    input_size = args->input_size;
    hidden_size = args->hidden_size;

    brgemm_arguments_input brgemm_arg;
    // brgemm_arguments_hidden brgemm_arg_1;

    worktile_t g;
    g.init(ei.get_local_linear_id());

    int start_m = ei.get_group(1) * wg_tile_m;
    int start_n = args->group_offset * wg_tile_n;
    int start_k = 0;
    int boundary_m, boundary_n, boundary_k;
    int local_pitch = ((hidden_size + wg_tile_n - 1) / wg_tile_n) * wg_tile_n;
    int wg_tile_k;
    mat_t mat_hidden;
    SW_BARRIER();

    int pitch_a, start_x_a, start_y_a;
    int pitch_b, start_x_b, start_y_b;
    boundary_n = (start_n + wg_tile_n) > hidden_size ? hidden_size
                                                     : (start_n + wg_tile_n);
    SW_BARRIER();
    init_acc_with_bias(
        args->B_ir_ptr,
        args->B_hr_ptr,
        &matAcc0,
        boundary_n,
        args->hidden_size,
        start_n + brgemm_op::get_matC_offset_x(g),
        start_m);
    init_acc_with_bias(
        args->B_iz_ptr,
        args->B_hz_ptr,
        &matAcc1,
        boundary_n,
        args->hidden_size,
        start_n + brgemm_op::get_matC_offset_x(g),
        start_m);
    init_acc_with_bias(
        args->B_hn_ptr,
        &matAcc2,
        boundary_n,
        args->hidden_size,
        start_n + brgemm_op::get_matC_offset_x(g),
        start_m);

    SW_BARRIER();
    // caculate Mat_Wi_ * Mat_X(t)
    PART_BRGEMM_CALL(
        0,
        &matAcc0,
        batch_size,
        input_size,
        hidden_size,
        args->layer_ptr,
        args->W_ir_ptr);
    // caculate Mat_Wh_ * Mat_h(t)

    PART_BRGEMM_CALL(
        0,
        &matAcc0,
        batch_size,
        hidden_size,
        hidden_size,
        args->hx_ptr,
        args->W_hr_ptr);
    tile_op::elemwise_op<matAcc_t, post_kind::sigmoid>(matAcc0);
    SW_BARRIER();
    MATC_STORE(0, args->reset_gate_ptr);

    PART_BRGEMM_CALL(
        0,
        &matAcc2,
        batch_size,
        hidden_size,
        hidden_size,
        args->hx_ptr,
        args->W_hn_ptr);

    MATC_STORE(2, args->hgate_2_ptr);
    SW_BARRIER();
    matAcc2.reg = matAcc2.reg * matAcc0.reg;

    init_acc_with_bias(
        args->B_in_ptr,
        &matAcc0,
        boundary_n,
        args->hidden_size,
        start_n + brgemm_op::get_matC_offset_x(g),
        start_m);
    PART_BRGEMM_CALL(
        0,
        &matAcc0,
        batch_size,
        input_size,
        hidden_size,
        args->layer_ptr,
        args->W_in_ptr);
    matAcc2.reg = matAcc2.reg + matAcc0.reg;
    tile_op::elemwise_op<matAcc_t, post_kind::tanh>(matAcc2);
    // caculate Mat_Wh_ * Mat_h(t)
    PART_BRGEMM_CALL(
        0,
        &matAcc1,
        batch_size,
        input_size,
        hidden_size,
        args->layer_ptr,
        args->W_iz_ptr);
    // caculate Mat_Wh_ * Mat_h(t)
    PART_BRGEMM_CALL(
        0,
        &matAcc1,
        batch_size,
        hidden_size,
        hidden_size,
        args->hx_ptr,
        args->W_hz_ptr);
    tile_op::elemwise_op<matAcc_t, post_kind::sigmoid>(matAcc1);
    SW_BARRIER();
    matC_base_desc.init(
        args->hx_ptr,
        boundary_n,
        boundary_m,
        hidden_size,
        start_n + ei.get_local_id(0) * sg_tile_n,
        start_m + ei.get_local_id(1) * sg_tile_m);
    mat_hidden.init_tdesc(matC_base_desc.get_tdesc());
    mat_hidden.template load<cache_hint::cached, cache_hint::cached>();
    SW_BARRIER();

    /// calculate reset gate: r_t = \sigma(X_t W_ir + h_{t - 1} W_hr)
    SW_BARRIER();
    MATC_STORE(2, args->new_gate_ptr);
    SW_BARRIER();
    MATC_STORE(1, args->input_gate_ptr);

    /// calculate h_t = (1 - z_t) n_t + z_t h_{t - 1} NOTICE z_t in Acc1, n_t in
    /// Acc0
    matAcc2.reg =
        matAcc2.reg * (1 - matAcc1.reg) + matAcc1.reg * mat_hidden.reg;
    MATC_STORE(2, args->cell_out_ptr);
    if (args->is_final_cell) {
      MATC_STORE(2, args->layer_output);
    }
    if (args->is_first_cell) {
      tile_op::elemwise_cvt<matAcc_t, mat_t>(&matAcc0, &mat_hidden);
      MATC_STORE(0, args->h0);
    }
    if (args->mask_ptr != nullptr) {
      mask_t drop_mask;
      mask_desc_t mask_base_desc;
      mask_base_desc.init(
          args->mask_ptr,
          boundary_n,
          boundary_m,
          hidden_size,
          start_n + ei.get_local_id(0) * sg_tile_n,
          start_m + ei.get_local_id(1) * sg_tile_m);
      drop_mask.init_tdesc(mask_base_desc.get_tdesc());
      drop_mask.template load<cache_hint::cached, cache_hint::cached>();
      matAcc2.reg = matAcc2.reg * drop_mask.reg;
    }

    MATC_STORE(2, args->hidden_ptr);
    SW_BARRIER();
  }

  static void inline init_acc_with_bias(
      Act_T* bias_ptr,
      matAcc_t* matAcc_ptr,
      uint32_t boundary_n,
      uint32_t matrix_n,
      int start_n,
      int start_m) {
    /// if bias_add, init matAcc with bias 1D data.
    xetpp_tdescriptor bias_base_desc;
    xetpp_fill_tdesc<Act_T>(
        bias_base_desc.xetpp_format<uint32_t>(),
        bias_ptr,
        boundary_n,
        1,
        matrix_n,
        start_n,
        0);
    using bias_t = xetpp_tile_load_t<
        Act_T,
        matAcc_t::reg_tile_size_x,
        1,
        matAcc_t::block_size_x,
        1,
        mem_layout::row_major,
        mem_space::global,
        gpu_arch::Xe,
        reg_layout::tiled>;
    bias_t bias;
    bias.init_tdesc(bias_base_desc);
    bias.template load<cache_hint::cached, cache_hint::cached>();
    matAcc_ptr->init_value(&bias);
  }

  static void inline init_acc_with_bias(
      Act_T* bias1_ptr,
      Act_T* bias2_ptr,
      matAcc_t* matAcc_ptr,
      uint32_t boundary_n,
      uint32_t matrix_n,
      int start_n,
      int start_m) {
    /// if bias_add, init matAcc with bias 1D data.
    xetpp_tdescriptor bias1_base_desc;
    xetpp_fill_tdesc<Act_T>(
        bias1_base_desc.xetpp_format<uint32_t>(),
        bias1_ptr,
        boundary_n,
        1,
        matrix_n,
        start_n,
        0);

    xetpp_tdescriptor bias2_base_desc;
    xetpp_fill_tdesc<Act_T>(
        bias2_base_desc.xetpp_format<uint32_t>(),
        bias2_ptr,
        boundary_n,
        1,
        matrix_n,
        start_n,
        0);
    using bias_t = xetpp_tile_load_t<
        Act_T,
        matAcc_t::reg_tile_size_x,
        1,
        matAcc_t::block_size_x,
        1,
        mem_layout::row_major,
        mem_space::global,
        gpu_arch::Xe,
        reg_layout::tiled>;
    bias_t bias1;
    bias_t bias2;
    bias1.init_tdesc(bias1_base_desc);
    bias1.template load<cache_hint::cached, cache_hint::cached>();
    bias2.init_tdesc(bias2_base_desc);
    bias2.template load<cache_hint::cached, cache_hint::cached>();
    SW_BARRIER();
    bias1.reg = bias1.reg + bias2.reg;
    matAcc_ptr->init_value(&bias1);
  }
};

template <
    typename input_T,
    typename Act_T,
    uint32_t wg_tile_m_t,
    uint32_t wg_tile_n_t,
    uint32_t sg_tile_m_t,
    uint32_t sg_tile_n_t,
    uint32_t sg_tile_k_0_t,
    uint32_t sg_tile_k_1_t>
struct kernel_gru_cell_fusion {
  static constexpr uint32_t fused_op_wg_m = wg_tile_m_t;
  static constexpr uint32_t fused_op_wg_n = wg_tile_n_t;
  static constexpr uint32_t fused_op_sg_m = sg_tile_m_t;
  static constexpr uint32_t fused_op_sg_n = sg_tile_n_t;
  static constexpr uint32_t fused_op_sg_k_0 = sg_tile_k_0_t;
  static constexpr uint32_t fused_op_sg_k_1 = sg_tile_k_1_t;
  using fused_cell_op = gru_cell<
      input_T,
      Act_T,
      fused_op_wg_m,
      fused_op_wg_n,
      fused_op_sg_m,
      fused_op_sg_n,
      fused_op_sg_k_0,
      fused_op_sg_k_1,
      mem_layout::row_major,
      mem_layout::row_major,
      mem_layout::col_major,
      mem_layout::row_major,
      mem_space::global,
      mem_space::global,
      mem_space::global,
      mem_space::global,
      mem_space::local, // TODO: support slm
      0>;

  /// @brief
  /// @param ei
  /// input
  /// @param layer_ptr        input from previous layer i.e X_t
  /// @param hx_ptr           hx_ptr input i.e. h_{0}  shape = layer_size x
  /// batch_size x hidden_size weights
  /// @param W_ir_ptr         weights with input of reset gate,
  /// (input_weight_size, hidden_weight_size, ...)
  /// @param W_hr_ptr         weights with hidden input of reset gate, shape =
  /// layer_size x hidden_weight_size
  /// @param W_iz_ptr         weights with input of input gate,
  /// (input_weight_size, hidden_weight_size, ...)
  /// @param W_hz_ptr         weights with hidden input of input gate, shape =
  /// layer_size x hidden_weight_size
  /// @param W_in_ptr         weights with input of new gate,
  /// (input_weight_size, hidden_weight_size, ...)
  /// @param W_hn_ptr         weights with hidden input of new gate, shape =
  /// layer_size x hidden_weight_size output
  /// @param layer_out_ptr    the last cell per layer output, shape = layer_size
  /// x batch_size x hidden_size
  /// @param hidden_out_ptr   the last layer output for per gru cell, shape =
  /// sequence_size x batch_size x hidden_size workspace:
  /// @param reset_gate_ptr   reset gate output
  /// @param input_gate_ptr   input gate output
  /// @param new_gate_ptr     new gate output
  /// @param hgate_2_ptr      hgate_2_ptr i.e h_{t - 1} W_hn
  /// @param workspace_ptr    all of the output per gru cell

  static void inline call(
      xetpp_exec_item<3> ei,
      input_T* layer_ptr,
      input_T* hx_ptr,
      input_T* i_weights,
      input_T* h_weights,
      Act_T* i_biases,
      Act_T* h_biases,
      input_T* layer_out_ptr,
      input_T* hidden_out_ptr,
      Act_T* mask_ptr,
      input_T* reset_gate_ptr,
      input_T* input_gate_ptr,
      input_T* new_gate_ptr,
      input_T* hgate_2_ptr,
      input_T* workspace_ptr,
      int batch_size,
      int input_size,
      int hidden_size,
      int sequence_size,
      int layer_size,
      int seq_idx,
      int layer_idx) {
    seq_idx = seq_idx - layer_idx;
    if (0 <= seq_idx && seq_idx < sequence_size) {
      fwd_config_t<input_T, Act_T> args;
      int input_weight_size = input_size * hidden_size;
      int hidden_weight_size = hidden_size * hidden_size;
      int hidden_io_size = batch_size * hidden_size;
      int layer_input_size = batch_size * input_size;

      int workspace_base_offset =
          layer_idx * (sequence_size + 1) * hidden_io_size;

      args.input_size = input_size;
      args.batch_size = batch_size;
      args.hidden_size = hidden_size;
      args.hx_ptr = (input_T*)(hx_ptr);
      args.layer_output = (input_T*)layer_out_ptr;
      args.is_final_layer = layer_idx == layer_size - 1;

      args.is_final_cell = seq_idx == sequence_size - 1;
      args.is_first_cell = seq_idx == 0;
      args.hx_ptr = args.is_first_cell
          ? (input_T*)(hx_ptr)
          : (input_T*)workspace_ptr + seq_idx * hidden_io_size;
      args.h0 = (input_T*)workspace_ptr;
      args.hidden_ptr =
          (input_T*)((input_T*)hidden_out_ptr + seq_idx * hidden_io_size);
      args.layer_ptr =
          (input_T*)((input_T*)layer_ptr + seq_idx * layer_input_size);
      args.W_ir_ptr = (input_T*)(i_weights);
      args.W_hr_ptr = (input_T*)(h_weights);
      args.W_iz_ptr = (input_T*)(i_weights) + input_weight_size;
      args.W_hz_ptr = (input_T*)(h_weights) + hidden_weight_size;
      args.W_in_ptr = (input_T*)(i_weights) + 2 * input_weight_size;
      args.W_hn_ptr = (input_T*)(h_weights) + 2 * hidden_weight_size;
      args.B_ir_ptr = (Act_T*)(i_biases);
      args.B_iz_ptr = (Act_T*)(i_biases) + hidden_size;
      args.B_in_ptr = (Act_T*)(i_biases) + hidden_size * 2;
      args.B_hr_ptr = (Act_T*)(h_biases);
      args.B_hz_ptr = (Act_T*)(h_biases) + hidden_size;
      args.B_hn_ptr = (Act_T*)(h_biases) + hidden_size * 2;
      args.reset_gate_ptr =
          (input_T*)((input_T*)reset_gate_ptr + seq_idx * hidden_io_size);
      args.input_gate_ptr =
          (input_T*)((input_T*)input_gate_ptr + seq_idx * hidden_io_size);
      args.new_gate_ptr =
          (input_T*)((input_T*)new_gate_ptr + seq_idx * hidden_io_size);
      args.hgate_2_ptr =
          (input_T*)((input_T*)hgate_2_ptr + seq_idx * hidden_io_size);
      args.cell_out_ptr =
          (input_T*)((input_T*)workspace_ptr + (seq_idx + 1) * hidden_io_size);
      args.mask_ptr = mask_ptr != nullptr
          ? (Act_T*)mask_ptr + seq_idx * hidden_io_size
          : nullptr;
      SW_BARRIER();
      for (unsigned j = 0; j < (hidden_size + wg_tile_n_t - 1) / wg_tile_n_t;
           ++j) {
        args.group_offset = j;
        fused_cell_op::call(ei, &args);
        __esimd_barrier();
      }
      SW_BARRIER();
    }
  }
};

/// @brief
/// inputs
/// @param layer_ptr
/// @param hx_ptr
/// weights
/// @param W_ir_ptr
/// @param W_hr_ptr
/// @param W_iz_ptr
/// @param W_hz_ptr
/// @param W_in_ptr
/// @param W_hn_ptr
/// outputs
/// @param layer_out_ptr
/// @param hidden_out_ptr
/// workspaces
/// @param reset_gate_ptr
/// @param input_gate_ptr
/// @param new_gate_ptr
/// @param hgate_2_ptr
/// @param workspace_ptr
/// params
/// @param batch_size
/// @param input_size
/// @param hidden_size
/// @param sequence_size
/// @param layer_size
/// @param Queue sycl::queue
/// @return
template <typename gru_config_t>
void gru_forward_impl(
    void* layer_ptr,
    void* hx_ptr,
    void* i_weights,
    void* h_weights,
    void* i_biases,
    void* h_biases,
    void* layer_out_ptr,
    void* hidden_out_ptr,
    void* mask_ptr,
    void* dropout_buffer,
    void* workspace_ptr,
    void* reset_gate_ptr,
    void* input_gate_ptr,
    void* new_gate_ptr,
    void* hgate_2_ptr,
    int batch_size,
    int input_size,
    int hidden_size,
    int sequence_size,
    int layer_size,
    cl::sycl::queue& Queue) {
  size_t M = batch_size;
  size_t N = hidden_size;
  using input_T = gru_config_t::input_T;
  using Act_T = gru_config_t::Act_T;
  constexpr uint32_t wg_tile_m = gru_config_t::wg_tile_m;
  constexpr uint32_t wg_tile_n = gru_config_t::wg_tile_n;
  constexpr uint32_t sg_tile_m = gru_config_t::sg_tile_m;
  constexpr uint32_t sg_tile_n = gru_config_t::sg_tile_n;
  constexpr uint32_t sg_tile_k_0 = gru_config_t::sg_tile_k_0;
  constexpr uint32_t sg_tile_k_1 = gru_config_t::sg_tile_k_1;
  using gru_op = kernel_gru_cell_fusion<
      input_T,
      Act_T,
      wg_tile_m,
      wg_tile_n,
      sg_tile_m,
      sg_tile_n,
      sg_tile_k_0,
      sg_tile_k_1>;
  const int num_layers = layer_size;
  cl::sycl::range<3> GroupRange = {
      num_layers,
      (M + wg_tile_m - 1) / wg_tile_m,
      // (H + wg_tile_n - 1) / wg_tile_n
      1};
  cl::sycl::range<3> LocalRange{
      1,
      (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
      (wg_tile_n + sg_tile_n - 1) / sg_tile_n};
  cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

  // launch kernels
  for (int seq = 0; seq < sequence_size + layer_size - 1; seq++) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      cgh.parallel_for(Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
        xetpp_exec_item ei(item);
        int layer_idx = ei.get_group(2);
        int hidden_io_size = batch_size * hidden_size;
        int input_weight_offset = layer_idx <= 1
            ? layer_idx * input_size * hidden_size * 3
            : input_size * hidden_size * 3 +
                (layer_idx - 1) * hidden_size * hidden_size * 3;
        int hidden_weight_offset = layer_idx * hidden_size * hidden_size * 3;
        int bias_offset = layer_idx * hidden_size * 3;
        int gates_base_offset = layer_idx * (sequence_size)*hidden_io_size;
        int workspace_base_offset =
            layer_idx * (sequence_size + 1) * hidden_io_size;
        gru_op::call(
            ei,
            layer_idx != 0 ? (input_T*)dropout_buffer +
                    (layer_idx - 1) * sequence_size * hidden_io_size
                           : (input_T*)layer_ptr, /* inputs*/
            (input_T*)hx_ptr + layer_idx * hidden_size * batch_size, /* hidden*/
            (input_T*)i_weights + input_weight_offset, /*weights*/
            (input_T*)h_weights + hidden_weight_offset, /*weights*/
            (Act_T*)i_biases + bias_offset, /*Bias*/
            (Act_T*)h_biases + bias_offset, /*Bias*/
            (input_T*)layer_out_ptr +
                layer_idx * hidden_size * batch_size, /*hn_out*/
            layer_idx == layer_size - 1
                ? (input_T*)hidden_out_ptr
                : (input_T*)dropout_buffer + gates_base_offset, /*output*/
            layer_idx < layer_size - 1 ? (Act_T*)mask_ptr + gates_base_offset
                                       : nullptr, /*mask*/
            (input_T*)reset_gate_ptr + gates_base_offset,
            (input_T*)input_gate_ptr + gates_base_offset,
            (input_T*)new_gate_ptr + gates_base_offset,
            (input_T*)hgate_2_ptr + gates_base_offset,
            (input_T*)workspace_ptr + workspace_base_offset, /*workspace*/
            batch_size,
            layer_idx == 0 ? input_size : hidden_size,
            hidden_size,
            sequence_size,
            layer_size,
            seq,
            layer_idx);
      });
    };
    DPCPP_Q_SUBMIT(Queue, cgf);
  }
}

} // namespace xetla
} // namespace xpu