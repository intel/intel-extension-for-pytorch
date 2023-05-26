#pragma once

#include <utils/DPCPP.h>
#include "../xetla.h"
#include "kernel_attr.h"

namespace xpu {
namespace xetla {

template <typename T, typename Act_T>
struct bpk_config_t {
  int input_size;
  int hidden_size;
  int batch_size;
  int sequence_length = 1;
  T* err_ptr0 =
      nullptr; /// initial err grads for backward progagation input shape =
               /// layer_size x sequence_length x batch_size x hidden_size
  T* err_ptr1 = nullptr;
  T* layer_ptr = nullptr; /// layer input inside forward procession shape =
                          /// sequence_length x batch_size x input_size
  T* hidden_ptr =
      nullptr; /// hidden outputs sequenece inside forward procession shape =
               /// layer_size x sequence_length x batch_size x hidden_size
  T* w_i_ptr = nullptr;
  T* w_h_ptr = nullptr;
  Act_T* bias0_ptr = nullptr;
  Act_T* bias1_ptr = nullptr;
  uint32_t slm_addr = 0;
};

#define BPK_CONFIG_SETTING(id, m, k, n)                 \
  boundary_n_##id = (start_n_##id + wg_tile_n_##id) > n \
      ? n                                               \
      : (start_n_##id + wg_tile_n_##id);                \
  matrix_n_##id = n;                                    \
  start_x_b_##id = start_n_##id;                        \
  start_y_b_##id = start_k;                             \
  brgemm_arg_##id.inner_loop_count = (wg_tile_k + sg_tile_k - 1) / sg_tile_k;

#define BPK_BRGEMM_CALL(op_id, ptr_a, ptr_b)                          \
  brgemm_arg_##op_id.matA_base_desc.init(                             \
      {ptr_a},                                                        \
      {boundary_k, boundary_m, is_col_major_a ? matrix_m : matrix_k}, \
      {start_x_a, start_y_a});                                        \
  brgemm_arg_##op_id.matB_base_desc.init(                             \
      {ptr_b},                                                        \
      {boundary_n_##op_id, boundary_k, matrix_n_##op_id},             \
      {start_x_b_##op_id, start_y_b_##op_id});                        \
  brgemm_op_##op_id(g, matAcc_##op_id, brgemm_arg_##op_id);           \
  SW_BARRIER();

#define BPK_MATC_STORE_GLOBAL(id, ptr_c)            \
  matC_base_desc.init(                              \
      {ptr_c},                                      \
      {boundary_n_##id, boundary_m, matrix_n_##id}, \
      {start_n_##id, start_m});                     \
  epilogue##id(g, matAcc_##id, matC_base_desc, epilogue_args_##id);

#define BPK_DESC_INIT(id, ptr)            \
  bpi_desc.init(                          \
      {ptr},                              \
      {boundary_m, boundary_k, matrix_m}, \
      {start_y_a + tile_offset_m, start_x_a + 0 * sg_tile_k});

#define BIAS_STORE_GLOBAL(id, ptr)                         \
  bias_desc.init(                                          \
      {ptr},                                               \
      {boundary_m, 1, matrix_m},                           \
      {start_m + brgemm_t_##id::get_matC_offset_y(g), 0}); \
  bias_payload.init(bias_desc);                            \
  tile_store(bias_##id, bias_payload);                     \
  SW_BARRIER();

#define BIAS_REDUCE(id)                                                \
  for (int j = 0; j < block_x_num; j++) {                              \
    for (int l = 0; l < block_y_num; l++) {                            \
      for (int k = 0; k < matA_bpi_t::block_size_y; k++) {             \
        bias_##id.reg.xetla_select<matA_bpi_t::block_size_x, 1>(       \
            j * matA_bpi_t::block_size_x) +=                           \
            matBPI_##id.reg.xetla_select<matA_bpi_t::block_size_x, 1>( \
                matA_bpi_t::block_elems * (j + l * block_x_num) +      \
                k * matA_bpi_t::block_size_x);                         \
      }                                                                \
    }                                                                  \
  }

template <
    typename T,
    typename Act_T,
    uint32_t wg_tile_n_0,
    uint32_t wg_tile_n_1,
    uint32_t wg_tile_m,
    uint32_t sg_tile_n_0,
    uint32_t sg_tile_n_1,
    uint32_t sg_tile_m,
    uint32_t sg_tile_k,
    mem_layout layout_input = mem_layout::row_major,
    mem_layout layout_hidden = mem_layout::row_major,
    mem_layout layout_err = mem_layout::col_major,
    mem_layout layout_weight = mem_layout::row_major,
    mem_layout layout_bias = mem_layout::row_major,
    mem_space mem_loc_input = mem_space::global,
    mem_space mem_loc_hidden = mem_space::global,
    mem_space mem_loc_err = mem_space::global,
    mem_space mem_loc_weight = mem_space::global,
    mem_space mem_loc_bias = mem_space::global,
    mem_space mem_loc_bpi = mem_space::global,
    uint32_t periodic_sync_interval = 0>
struct gru_layer_bpk {
  static constexpr uint32_t prefetch_distance = 3;
  using perf_tuning_knob =
      perf_tuning_knob_t<16, prefetch_distance, periodic_sync_interval>;

  using compute_attr = compute_attr_t<T, T, Act_T>;
  using compute_policy =
      compute_policy_default_xmx<compute_attr, perf_tuning_knob, gpu_arch::Xe>;

  static constexpr uint32_t tg_size_x =
      (wg_tile_n_0 + sg_tile_n_0 - 1) / sg_tile_n_0;
  static constexpr uint32_t tg_size_y = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
  // using worktile_t = xetla_worktile_t<tg_size_x * tg_size_y>;
  using tile_shape_0 = tile_shape_t<
      wg_tile_n_0, // workgroup size in N dim
      wg_tile_m, //	workgroup size in M dim
      sg_tile_n_0, //	subgroup size in N dim
      sg_tile_m>; //	subgroup size in M dim
  using tile_shape_1 =
      tile_shape_t<wg_tile_n_1, wg_tile_m, sg_tile_n_1, sg_tile_m>;
  static constexpr bool is_col_major_b_0 =
      layout_hidden == mem_layout::col_major;
  static constexpr bool is_col_major_b_1 =
      layout_input == mem_layout::col_major;
  static constexpr bool is_col_major_a = layout_err == mem_layout::col_major;

  using mem_desc_weight_t = mem_desc_t<T, layout_weight, mem_loc_weight>;
  using mem_desc_bias_t = mem_desc_t<Act_T, layout_bias, mem_loc_bias>;
  using mem_desc_err_t = mem_desc_t<T, layout_err, mem_loc_err>;
  using mem_desc_bpi_t = mem_desc_t<T, mem_layout::row_major, mem_loc_bpi>;
  using mem_desc_input_t = mem_desc_t<T, layout_input, mem_loc_input>;
  using mem_desc_hidden_t = mem_desc_t<T, layout_hidden, mem_loc_hidden>;

  using epilogue0_t = epilogue_t<
      epilogue_policy_tile_op<
          subgroup::chained_tile_op_t<>,
          result_overwrite,
          gpu_arch::Xe>,
      tile_shape_0,
      mem_desc_weight_t>;
  using epilogue1_t = epilogue_t<
      epilogue_policy_tile_op<
          subgroup::chained_tile_op_t<>,
          result_overwrite,
          gpu_arch::Xe>,
      tile_shape_1,
      mem_desc_weight_t>;
  using epilogue_args_0_t = typename epilogue0_t::arguments_t;
  using epilogue_args_1_t = typename epilogue1_t::arguments_t;

  using brgemm_t_0 =
      brgemm_t<compute_policy, tile_shape_0, mem_desc_err_t, mem_desc_input_t>;
  using brgemm_t_1 =
      brgemm_t<compute_policy, tile_shape_1, mem_desc_err_t, mem_desc_hidden_t>;

  using worker_scope_t = typename brgemm_t_0::work_group_t;

  using brgemm_arguments_hidden = typename brgemm_t_0::arguments_t;
  using brgemm_arguments_input = typename brgemm_t_1::arguments_t;

  using matAcc_t0 = typename brgemm_t_0::matAcc_t;
  using matAcc_t1 = typename brgemm_t_1::matAcc_t;

  using matAcc_desc_t0 = typename brgemm_t_0::matAcc_tile_desc_t;
  using matAcc_desc_t1 = typename brgemm_t_1::matAcc_tile_desc_t;

  using block_attr =
      get_load_block_size_auto<T, sg_tile_m, sg_tile_k, gpu_arch::Xe>;

  using matA_tile_desc_t = tile_desc_t<
      sg_tile_m,
      sg_tile_k,
      block_attr::block_size_x,
      block_attr::block_size_y,
      reg_layout::tiled>;
  using matA_load_0_t = tile_t<T, matA_tile_desc_t>;
  using matA_payload_0_t = mem_payload_t<
      T,
      matA_tile_desc_t,
      msg_type_v<matA_tile_desc_t, mem_space::global>,
      mem_layout::row_major,
      mem_space::global,
      gpu_arch::Xe>;

  using matA_bpi_t = tile_t<Act_T, matA_tile_desc_t>;

  using matC_payload_t0 = mem_payload_t<
      T,
      matAcc_desc_t0,
      msg_type_v<matAcc_desc_t0, mem_loc_weight>,
      layout_weight,
      mem_loc_weight,
      gpu_arch::Xe>;
  using matC_t0 = tile_t<T, matAcc_desc_t0>;
  using matC_payload_t1 = mem_payload_t<
      T,
      matAcc_desc_t1,
      msg_type_v<matAcc_desc_t1, mem_loc_weight>,
      layout_weight,
      mem_loc_weight,
      gpu_arch::Xe>;
  using matC_t1 = tile_t<T, matAcc_desc_t1>;

  using bias_desc_t = tile_desc_t<
      matA_bpi_t::tile_size_x,
      1,
      matA_bpi_t::block_size_x,
      1,
      reg_layout::tiled>;
  using bias_t = tile_t<Act_T, bias_desc_t>;
  using bias_payload_t = mem_payload_t<
      Act_T,
      bias_desc_t,
      msg_type_v<bias_desc_t, mem_loc_bias>,
      layout_bias,
      mem_loc_bias,
      gpu_arch::Xe>;

  using prefetch_t = prefetch_payload_t<
      T,
      tile_desc_t<matA_load_0_t::tile_size_x, matA_load_0_t::tile_size_y, 1, 1>,
      mem_layout::row_major,
      mem_space::global,
      1, /*tg_size_x=1*/
      gpu_arch::Xe>;
  static constexpr tdesc_update_dir load_update_config =
      tdesc_update_dir::y_dir;

  static void inline call(xetla_exec_item<3> ei, bpk_config_t<T, Act_T>* args) {
    int batch_size, input_size, hidden_size;
    batch_size = args->batch_size;
    input_size = args->input_size;
    hidden_size = args->hidden_size;

    matC_t0 matC_0; /// for W_h* grads
    matC_t1 matC_1; /// for W_i* grads
    matC_payload_t0 matC_payload_0; /// for W_h* grads
    matC_payload_t1 matC_payload_1; /// for W_i* grads

    matA_load_0_t matBPI_load_0, matBPI_load_1; /// load data src0
    matA_payload_0_t matBPI_payload_0, matBPI_payload_1;

    matA_bpi_t matBPI_0, matBPI_1; /// dst = cvt<src0 * src1>
    mem_desc_weight_t matC_base_desc;
    mem_desc_bpi_t bpi_desc;
    mem_desc_bias_t bias_desc;

    brgemm_t_0 brgemm_op_0;
    brgemm_t_1 brgemm_op_1;
    brgemm_arguments_hidden brgemm_arg_0;
    brgemm_arguments_input brgemm_arg_1;

    epilogue0_t epilogue0;
    epilogue1_t epilogue1;

    epilogue_args_0_t epilogue_args_0{};
    epilogue_args_1_t epilogue_args_1{};

    matAcc_t0 matAcc_0;
    matAcc_t1 matAcc_1;

    bias_t bias_0, bias_1;
    bias_payload_t bias_payload;

    int matrix_n_0, start_x_b_0, start_y_b_0;
    int matrix_n_1, start_x_b_1, start_y_b_1;
    int start_n_0 = ei.get_group(2) * wg_tile_n_0;
    int start_n_1 = ei.get_group(2) * wg_tile_n_1;
    int start_m = ei.get_group(1) * wg_tile_m;
    int start_k = 0;
    int boundary_n_0, boundary_n_1, boundary_m, boundary_k;
    int wg_tile_k;

    wg_tile_k = batch_size;
    boundary_k = wg_tile_k;

    BPK_CONFIG_SETTING(0, 3 * hidden_size, batch_size, hidden_size);
    BPK_CONFIG_SETTING(1, 3 * hidden_size, batch_size, input_size);

    boundary_m = (start_m + wg_tile_m) > 3 * hidden_size
        ? 3 * hidden_size
        : (start_m + wg_tile_m);
    int matrix_m = 3 * hidden_size;
    int matrix_k = batch_size;
    int start_x_a = start_k;
    int start_y_a = start_m;

    int32_t tile_offset_m = ei.get_local_id(1) * sg_tile_m;
    int offset_x_a = is_col_major_a ? tile_offset_m : 0;
    int offset_y_a = is_col_major_a ? 0 : tile_offset_m;

    int io_size = batch_size * hidden_size;
    int gate_nums = 3;
    int layer_input_size = batch_size * input_size;
    int seq_len = args->sequence_length;
    matAcc_0.init(0);
    matAcc_1.init(0);
    int block_x_num = matA_bpi_t::tile_size_x / matA_bpi_t::block_size_x;
    int block_y_num = matA_bpi_t::tile_size_y / matA_bpi_t::block_size_y;
    bias_0.init(0);
    bias_1.init(0);
    matBPI_0.init(0);
    matBPI_1.init(0);
    worker_scope_t g(ei.get_local_linear_id());

    for (unsigned seq_id = 0; seq_id < seq_len; ++seq_id) {
      BPK_DESC_INIT(
          0, args->err_ptr1 + (seq_len - 1 - seq_id) * io_size * gate_nums);
      matBPI_payload_0.init(bpi_desc);
      prefetch_t prefetch0(bpi_desc);
      prefetch0.template update_tdesc<load_update_config>(sg_tile_k);
      tile_load(matBPI_load_0, matBPI_payload_0);
      matBPI_payload_0.template update_tdesc<load_update_config>(sg_tile_k);
      BPK_DESC_INIT(
          1, args->err_ptr0 + (seq_len - 1 - seq_id) * io_size * gate_nums);
      matBPI_payload_1.init(bpi_desc);
      prefetch_t prefetch1(bpi_desc);
      prefetch1.template update_tdesc<load_update_config>(sg_tile_k);
      tile_load(matBPI_load_1, matBPI_payload_1);
      matBPI_payload_1.template update_tdesc<load_update_config>(sg_tile_k);
      matBPI_1.reg +=
          xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_0.reg);
      SW_BARRIER();
      matBPI_0.reg +=
          xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_1.reg);
      SW_BARRIER();
#pragma unroll
      for (int i = 0; i < prefetch_distance; i++) {
        tile_prefetch(prefetch0);
        prefetch0.template update_tdesc<load_update_config>(sg_tile_k);
        tile_prefetch(prefetch1);
        prefetch1.template update_tdesc<load_update_config>(sg_tile_k);
      }
      for (int i = 1; i < (wg_tile_k + sg_tile_k - 1) / sg_tile_k; i++) {
        tile_load(matBPI_load_0, matBPI_payload_0);
        matBPI_payload_0.template update_tdesc<load_update_config>(sg_tile_k);
        tile_prefetch(prefetch0);
        prefetch0.template update_tdesc<load_update_config>(sg_tile_k);
        // MAT_LOAD_GLOBAL(0, args->err_ptr1 + (seq_len - 1 - seq_id) * io_size
        // * gate_nums);    /// err
        matBPI_1.reg +=
            xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_0.reg);
        SW_BARRIER();
        tile_load(matBPI_load_1, matBPI_payload_1);
        matBPI_payload_1.template update_tdesc<load_update_config>(sg_tile_k);
        tile_prefetch(prefetch1);
        prefetch1.template update_tdesc<load_update_config>(sg_tile_k);
        // MAT_LOAD_GLOBAL(0, args->err_ptr0 + (seq_len - 1 - seq_id) * io_size
        // * gate_nums);    /// err
        matBPI_0.reg +=
            xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_1.reg);
        SW_BARRIER();
      }

      BPK_BRGEMM_CALL(
          1,
          args->err_ptr1 + (seq_len - 1 - seq_id) * io_size * gate_nums,
          args->layer_ptr + (seq_len - 1 - seq_id) * layer_input_size);

      /// GEMM_0
      BPK_BRGEMM_CALL(
          0,
          args->err_ptr0 + (seq_len - 1 - seq_id) * io_size * gate_nums,
          args->hidden_ptr + (seq_len - 1 - seq_id) * io_size);
    }
    BIAS_REDUCE(0);
    BIAS_REDUCE(1);

    BIAS_STORE_GLOBAL(0, args->bias0_ptr);
    // SW_BARRIER();
    BIAS_STORE_GLOBAL(1, args->bias1_ptr);
    // SW_BARRIER();
    BPK_MATC_STORE_GLOBAL(1, args->w_i_ptr);
    SW_BARRIER();
    BPK_MATC_STORE_GLOBAL(0, args->w_h_ptr);
    SW_BARRIER();
  }
};

template <
    typename input_T,
    typename Act_T,
    uint32_t wg_tile_n_0_t,
    uint32_t wg_tile_n_1_t,
    uint32_t wg_tile_m_t,
    uint32_t sg_tile_n_0_t,
    uint32_t sg_tile_n_1_t,
    uint32_t sg_tile_m_t,
    uint32_t sg_tile_k_t>
struct perf_kernel_xcoder_gru_bpk {
  /// @brief
  /// @param ei
  /// @param err_ptr      err ptr
  /// @param layer_ptr    original layer_input      shape = sequence_length x
  /// batch_size x input_size
  /// @param hidden_ptr         hidden outputs per gru cell     shape =
  /// layer_size x sequence_size x batch_size x hidden_size
  /// @param w_i_ptr     output w i grad array shape = [hidden_]
  /// @param w_h_ptr     output w h grad
  /// @param bias_ptr    output shared grad for reset gate bias 0/1
  /// @param batch_size
  /// @param input_size
  /// @param hidden_size
  /// @param sequence_length
  /// @param layer_size
  static void inline run(
      xetla_exec_item<3> ei,
      input_T* err0_ptr,
      input_T* err1_ptr,
      input_T* layer_ptr,
      input_T* hidden_ptr,
      input_T* w_i_ptr, /// [r, z, n]
      input_T* w_h_ptr, /// [r, z, n]
      Act_T* bias0_ptr,
      Act_T* bias1_ptr,
      int batch_size,
      int input_size,
      int hidden_size,
      int sequence_length,
      int layer_size) {
    constexpr uint32_t fused_op_wg_n_0 = wg_tile_n_0_t;
    constexpr uint32_t fused_op_wg_n_1 = wg_tile_n_1_t;
    constexpr uint32_t fused_op_wg_m = wg_tile_m_t;
    constexpr uint32_t fused_op_sg_n_0 = sg_tile_n_0_t;
    constexpr uint32_t fused_op_sg_n_1 = sg_tile_n_1_t;
    constexpr uint32_t fused_op_sg_m = sg_tile_m_t;
    constexpr uint32_t fused_op_sg_k = sg_tile_k_t;
    using fused_op_0 = gru_layer_bpk<
        input_T,
        Act_T,
        fused_op_wg_n_0,
        fused_op_wg_n_1,
        fused_op_wg_m,
        fused_op_sg_n_0,
        fused_op_sg_n_1,
        fused_op_sg_m,
        fused_op_sg_k>;
    using fused_op_1 = gru_layer_bpk<
        input_T,
        Act_T,
        fused_op_wg_n_0,
        fused_op_wg_n_0,
        fused_op_wg_m,
        fused_op_sg_n_0,
        fused_op_sg_n_0,
        fused_op_sg_m,
        fused_op_sg_k>;
    bpk_config_t<input_T, Act_T> args;

    int layer_input_size = batch_size * input_size;
    int hidden_io_size = batch_size * hidden_size;
    int input_weight_size = input_size * hidden_size;
    int hidden_weight_size = hidden_size * hidden_size;
    int one_layer_size = hidden_io_size * sequence_length;
    int one_layer_io_size = hidden_io_size * (sequence_length + 1);
    int gate_nums = 3;

    // args.debug_ptr = debug_ptr;
    int layer_id = ei.get_group(0);
    args.sequence_length = sequence_length;

    if (layer_id != 0 && layer_id < layer_size) {
      // for (int layer_id = layer_size - 1; layer_id > 0; layer_id--) {
      args.batch_size = batch_size;
      args.hidden_size = hidden_size;
      args.input_size = hidden_size;
      args.slm_addr = 0;
      args.err_ptr0 = err0_ptr + layer_id * one_layer_size * gate_nums;
      args.err_ptr1 = err1_ptr + layer_id * one_layer_size * gate_nums;
      args.layer_ptr =
          hidden_ptr + (layer_id - 1) * one_layer_io_size + hidden_io_size;
      args.hidden_ptr = hidden_ptr + layer_id * one_layer_io_size;
      args.w_i_ptr = layer_id < 1 ? w_i_ptr + layer_id * 3 * input_weight_size
                                  : w_i_ptr + 3 * input_weight_size +
              (layer_id - 1) * 3 * hidden_weight_size;
      args.w_h_ptr = w_h_ptr + layer_id * 3 * hidden_weight_size;
      args.bias0_ptr = bias0_ptr + layer_id * hidden_size * 3;
      args.bias1_ptr = bias1_ptr + layer_id * hidden_size * 3;
      SW_BARRIER();
      fused_op_1::call(ei, &args);
    } else {
      args.slm_addr = 0;
      args.err_ptr0 = err0_ptr;
      args.err_ptr1 = err1_ptr;
      args.layer_ptr = layer_ptr;
      args.hidden_ptr = hidden_ptr;
      args.w_i_ptr = w_i_ptr;
      args.w_h_ptr = w_h_ptr;
      args.bias0_ptr = bias0_ptr;
      args.bias1_ptr = bias1_ptr;
      args.batch_size = batch_size;
      args.hidden_size = hidden_size;
      args.input_size = input_size;
      SW_BARRIER();
      fused_op_0::call(ei, &args);
    }
  }
};

template <
    typename input_T,
    typename Act_T,
    uint32_t wg_tile_n_0_t,
    uint32_t wg_tile_n_1_t,
    uint32_t wg_tile_m_t,
    uint32_t sg_tile_n_0_t,
    uint32_t sg_tile_n_1_t,
    uint32_t sg_tile_m_t,
    uint32_t sg_tile_k_t>
struct kernel_xcoder_gru_bpk {
  /// @brief
  /// @param ei
  /// @param err_ptr      err ptr
  /// @param layer_ptr    original layer_input      shape = sequence_length x
  /// batch_size x input_size
  /// @param hidden_ptr         hidden outputs per gru cell     shape =
  /// layer_size x sequence_size x batch_size x hidden_size
  /// @param w_i_ptr     output w i grad array shape = [hidden_]
  /// @param w_h_ptr     output w h grad
  /// @param bias_ptr    output shared grad for reset gate bias 0/1
  /// @param batch_size
  /// @param input_size
  /// @param hidden_size
  /// @param sequence_length
  /// @param layer_size
  static void inline run(
      xetla_exec_item<3> ei,
      input_T* err0_ptr,
      input_T* err1_ptr,
      input_T* layer_ptr,
      input_T* hidden_ptr,
      input_T* w_i_ptr, /// [r, z, n]
      input_T* w_h_ptr, /// [r, z, n]
      Act_T* bias0_ptr,
      Act_T* bias1_ptr,
      int batch_size,
      int input_size,
      int hidden_size,
      int sequence_length,
      int layer_size) {
    constexpr uint32_t fused_op_wg_n_0 = wg_tile_n_0_t;
    constexpr uint32_t fused_op_wg_n_1 = wg_tile_n_1_t;
    constexpr uint32_t fused_op_wg_m = wg_tile_m_t;
    constexpr uint32_t fused_op_sg_n_0 = sg_tile_n_0_t;
    constexpr uint32_t fused_op_sg_n_1 = sg_tile_n_1_t;
    constexpr uint32_t fused_op_sg_m = sg_tile_m_t;
    constexpr uint32_t fused_op_sg_k = sg_tile_k_t;
    using fused_op_0 = gru_layer_bpk<
        input_T,
        Act_T,
        fused_op_wg_n_0,
        fused_op_wg_n_1,
        fused_op_wg_m,
        fused_op_sg_n_0,
        fused_op_sg_n_1,
        fused_op_sg_m,
        fused_op_sg_k>;
    using fused_op_1 = gru_layer_bpk<
        input_T,
        Act_T,
        fused_op_wg_n_0,
        fused_op_wg_n_0,
        fused_op_wg_m,
        fused_op_sg_n_0,
        fused_op_sg_n_0,
        fused_op_sg_m,
        fused_op_sg_k>;
    bpk_config_t<input_T, Act_T> args;
    int layer_input_size = batch_size * input_size;
    int hidden_io_size = batch_size * hidden_size;
    int input_weight_size = input_size * hidden_size;
    int hidden_weight_size = hidden_size * hidden_size;
    int one_layer_size = hidden_io_size * sequence_length;
    int one_layer_io_size = hidden_io_size * (sequence_length + 1);
    int gate_nums = 3;

    args.sequence_length = sequence_length;

    for (int layer_id = layer_size - 1; layer_id > 0; layer_id--) {
      args.batch_size = batch_size;
      args.hidden_size = hidden_size;
      args.input_size = hidden_size;
      args.slm_addr = 0;
      args.err_ptr0 = err0_ptr + layer_id * one_layer_size * gate_nums;
      args.err_ptr1 = err1_ptr + layer_id * one_layer_size * gate_nums;
      args.layer_ptr =
          hidden_ptr + (layer_id - 1) * one_layer_io_size + hidden_io_size;
      args.hidden_ptr = hidden_ptr + layer_id * one_layer_io_size;
      args.w_i_ptr = layer_id < 1 ? w_i_ptr + layer_id * 3 * input_weight_size
                                  : w_i_ptr + 3 * input_weight_size +
              (layer_id - 1) * 3 * hidden_weight_size;
      args.w_h_ptr = w_h_ptr + layer_id * 3 * hidden_weight_size;
      args.bias0_ptr = bias0_ptr + layer_id * hidden_size * 3;
      args.bias1_ptr = bias1_ptr + layer_id * hidden_size * 3;
      SW_BARRIER();
      fused_op_1::call(ei, &args);
    }
    args.slm_addr = 0;
    args.err_ptr0 = err0_ptr;
    args.err_ptr1 = err1_ptr;
    args.layer_ptr = layer_ptr;
    args.hidden_ptr = hidden_ptr;
    args.w_i_ptr = w_i_ptr;
    args.w_h_ptr = w_h_ptr;
    args.bias0_ptr = bias0_ptr;
    args.bias1_ptr = bias1_ptr;
    args.batch_size = batch_size;
    args.hidden_size = hidden_size;
    args.input_size = input_size;
    SW_BARRIER();
    fused_op_0::call(ei, &args);
  }
};
// extern "C"
template <typename gru_bpk_config_t>
void gru_backward_weight_impl(
    void* err0_ptr,
    void* err1_ptr,
    void* layer_ptr,
    void* hidden_ptr,
    void* w_i_ptr,
    void* w_h_ptr,
    void* bias0_ptr,
    void* bias1_ptr,
    int batch_size,
    int input_size,
    int hidden_size,
    int sequence_length,
    int layer_size,
    cl::sycl::queue& Queue) {
  size_t wg_tile_n_0 = gru_bpk_config_t::wg_tile_n_0;
  size_t wg_tile_n_1 = gru_bpk_config_t::wg_tile_n_1;
  size_t wg_tile_m = gru_bpk_config_t::wg_tile_m;
  size_t sg_tile_n_0 = gru_bpk_config_t::sg_tile_n_0;
  size_t sg_tile_n_1 = gru_bpk_config_t::sg_tile_n_1;
  size_t sg_tile_m = gru_bpk_config_t::sg_tile_m;
  size_t sg_tile_k = gru_bpk_config_t::sg_tile_k;

  using input = gru_bpk_config_t::input_T;
  using Act = gru_bpk_config_t::Act_T;

  const int num_layers = layer_size;
  size_t H = hidden_size;

  cl::sycl::range<3> GroupRange{
      num_layers,
      (3 * H + wg_tile_m - 1) / wg_tile_m,
      (H + wg_tile_n_0 - 1) / wg_tile_n_0};
  cl::sycl::range<3> LocalRange{
      1,
      (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
      (wg_tile_n_0 + sg_tile_n_0 - 1) / sg_tile_n_0};
  cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item ei(item);

      using xcoder_gru_bpk_op = perf_kernel_xcoder_gru_bpk<
          typename gru_bpk_config_t::input_T,
          typename gru_bpk_config_t::Act_T,
          gru_bpk_config_t::wg_tile_n_0,
          gru_bpk_config_t::wg_tile_n_1,
          gru_bpk_config_t::wg_tile_m,
          gru_bpk_config_t::sg_tile_n_0,
          gru_bpk_config_t::sg_tile_n_1,
          gru_bpk_config_t::sg_tile_m,
          gru_bpk_config_t::sg_tile_k>;

      xcoder_gru_bpk_op::run(
          ei,
          (input*)err0_ptr,
          (input*)err1_ptr,
          (input*)layer_ptr,
          (input*)hidden_ptr, /* inputs*/
          (input*)w_i_ptr,
          (input*)w_h_ptr, /*weights grads outputs*/
          (float*)bias0_ptr,
          (float*)bias1_ptr, /*bias grad outputs*/
          batch_size,
          input_size,
          hidden_size,
          sequence_length,
          layer_size);
    });
  };
  DPCPP_Q_SUBMIT(Queue, cgf);
}

} // namespace xetla
} // namespace xpu
