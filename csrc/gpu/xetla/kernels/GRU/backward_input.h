#pragma once

#include <utils/DPCPP.h>
#include "../xetla.h"
#include "kernel_attr.h"

namespace xpu {
namespace xetla {

template <typename T>
struct bpi_config_t {
  int input_size;
  int hidden_size;
  int batch_size;
  int sequence_length = 1;
  T* layer_err_ptr = nullptr;
  T* partial_err_ptr = nullptr;
  T* x_grad_ptr = nullptr;
  T* bpi0_ptr = nullptr;
  T* bpi1_ptr = nullptr;
  T* x0_grad_ptr = nullptr;
  T* hidden_ptr = nullptr;
  T* reset_gate_ptr = nullptr;
  T* input_gate_ptr = nullptr;
  T* new_gate_ptr = nullptr;
  T* hgate_2_ptr = nullptr;
  T* w_i_ptr = nullptr;
  T* w_h_ptr = nullptr;
  float* mask_ptr = nullptr; // mask matrix pointer
  uint32_t slm_addr = 0;
  float reserve;
  float dropout = 0;
};

#define BPI_CONFIG_SETTING(id, m, k, n)                 \
  boundary_n_##id = (start_n_##id + wg_tile_n_##id) > n \
      ? n                                               \
      : (start_n_##id + wg_tile_n_##id);                \
  matrix_n_##id = n;                                    \
  start_x_b_##id = start_n_##id;                        \
  start_y_b_##id = start_k;                             \
  brgemm_arg_##id.inner_loop_count =                    \
      (3 * wg_tile_k + sg_tile_k - 1) / sg_tile_k;

#define BPI_BRGEMM_CALL(op_id, ptr_b, ptr_a)                  \
  brgemm_arg_##op_id.matB_base_desc.init(                     \
      {ptr_b},                                                \
      {boundary_n_##op_id, 3 * boundary_k, matrix_n_##op_id}, \
      {start_x_b_##op_id, start_y_b_##op_id});                \
  brgemm_arg_##op_id.matA_base_desc.init(                     \
      {ptr_a},                                                \
      {3 * boundary_k, boundary_m, 3 * matrix_k},             \
      {start_x_a, start_y_a});                                \
  brgemm_op_##op_id(g, matAcc_##op_id, brgemm_arg_##op_id);   \
  SW_BARRIER();

#define BPI_MATC_STORE_GLOBAL(id, ptr_c, pitch)                   \
  matC_base_desc.init(                                            \
      {ptr_c},                                                    \
      {boundary_n_##id, boundary_m, matrix_n_##id},               \
      {start_n_##id + brgemm_t_##id::get_matC_offset_x(g),        \
       start_m + brgemm_t_##id::get_matC_offset_y(g)});           \
  matC_payload_##id.init(matC_base_desc);                         \
  elemwise_cvt<matC_t##id, matAcc_t##id>(matC_##id, matAcc_##id); \
  tile_store(matC_##id, matC_payload_##id);                       \
  SW_BARRIER();

#define MACC_INIT_LOAD(id, ptr_c, pitch)                                    \
  matC_base_desc.init(                                                      \
      {ptr_c},                                                              \
      {boundary_n_##id, boundary_m, matrix_n_##id},                         \
      {ei.get_group(2) * wg_tile_n_0 + brgemm_t_##id::get_matC_offset_x(g), \
       start_m + brgemm_t_##id::get_matC_offset_y(g)});                     \
  matC_payload_0.init(matC_base_desc);                                      \
  tile_load(matAcc_init, matC_payload_0);                                   \
  SW_BARRIER();

#define BPI_DESC_INIT(id, ptr)            \
  matC_base_desc.init(                    \
      {ptr},                              \
      {boundary_k, boundary_m, matrix_k}, \
      {lhs_start_k + offset_x_a, start_y_a + offset_y_a});

#define MAT_STORE_GLOBAL(id, ptr)                                     \
  matC_base_desc.init(                                                \
      {ptr},                                                          \
      {3 * boundary_k, boundary_m, 3 * matrix_k},                     \
      {lhs_start_k + offset_x_a + i * sg_tile_k + matrix_k * id,      \
       start_y_a + offset_y_a});                                      \
  matBPI_payload_g.init(matC_base_desc);                              \
  elemwise_cvt<matA_load_0_t, matA_bpi_t>(matBPI_store, matBPI_##id); \
  tile_store(matBPI_store, matBPI_payload_g);                         \
  SW_BARRIER();

#define DROPOUT(id, mask_ptr)                              \
  mask_desc.init(                                          \
      {mask_ptr},                                          \
      {boundary_n_##id, boundary_m, matrix_n_##id},        \
      {start_n_##id + brgemm_t_##id::get_matC_offset_x(g), \
       start_m + brgemm_t_##id::get_matC_offset_y(g)});    \
  mask_payload.init(mask_desc);                            \
  tile_load(mask_in, mask_payload);                        \
  matAcc_##id.reg = matAcc_##id.reg * mask_in.reg * args->reserve;

template <
    typename T,
    typename Act_T,
    uint32_t wg_tile_m,
    uint32_t wg_tile_n_0,
    uint32_t wg_tile_n_1,
    uint32_t sg_tile_m,
    uint32_t sg_tile_n_0,
    uint32_t sg_tile_n_1,
    uint32_t sg_tile_k,
    mem_layout layout_grad = mem_layout::row_major,
    mem_layout layout_weight = mem_layout::row_major,
    mem_space mem_loc_grad = mem_space::global,
    mem_space mem_loc_weight = mem_space::global,
    mem_space mem_loc_bpi = mem_space::global,
    uint32_t periodic_sync_interval = 0>
struct gru_layer_bpi {
  static constexpr uint32_t prefetch_distance = 3;
  using perf_tuning_knob = perf_tuning_knob_t<sg_tile_k, prefetch_distance>;
  using compute_attr = compute_attr_t<T, T, Act_T>;
  using compute_policy =
      compute_policy_default_xmx<compute_attr, perf_tuning_knob, gpu_arch::Xe>;

  using tile_shape_0 = tile_shape_t<
      wg_tile_n_0, // workgroup size in N dim
      wg_tile_m, //	workgroup size in M dim
      sg_tile_n_0, //	subgroup size in N dim
      sg_tile_m>; //	subgroup size in M dim
  using tile_shape_1 =
      tile_shape_t<wg_tile_n_1, wg_tile_m, sg_tile_n_1, sg_tile_m>;

  static constexpr bool is_col_major_a = layout_grad == mem_layout::col_major;

  static constexpr bool is_col_major_b = layout_weight == mem_layout::col_major;

  using mem_desc_err_t = mem_desc_t<T, layout_grad, mem_loc_grad>;
  using mem_desc_weight_t = mem_desc_t<T, layout_weight, mem_loc_weight>;
  using mem_desc_out_t = mem_desc_t<T, layout_grad, mem_loc_bpi>;
  using mem_desc_mask_t =
      mem_desc_t<Act_T, mem_layout::row_major, mem_space::global>;

  using brgemm_t_0 =
      brgemm_t<compute_policy, tile_shape_0, mem_desc_err_t, mem_desc_weight_t>;
  using brgemm_t_1 =
      brgemm_t<compute_policy, tile_shape_1, mem_desc_err_t, mem_desc_weight_t>;

  using worker_scope_t = typename brgemm_t_0::work_group_t;

  using brgemm_arguments_hidden = typename brgemm_t_0::arguments_t;
  using brgemm_arguments_input = typename brgemm_t_1::arguments_t;

  using matAcc_t0 = typename brgemm_t_0::matAcc_t;
  using matAcc_t1 = typename brgemm_t_1::matAcc_t;

  using matAcc_tile_desc_t0 = typename brgemm_t_0::matAcc_tile_desc_t;
  using matAcc_tile_desc_t1 = typename brgemm_t_1::matAcc_tile_desc_t;

  using matA_tile_desc_t = tile_desc_t<
      brgemm_t_0::matA_t::tile_size_x,
      brgemm_t_0::matA_t::tile_size_y,
      brgemm_t_0::matA_t::block_size_x,
      brgemm_t_0::matA_t::block_size_y,
      reg_layout::tiled>;

  using matA_payload_global_t = mem_payload_t<
      T,
      matA_tile_desc_t,
      msg_type_v<matA_tile_desc_t, mem_loc_grad>,
      layout_grad,
      mem_loc_grad,
      gpu_arch::Xe>;

  using matA_load_0_t = tile_t<T, matA_tile_desc_t>;

  using prefetch_t = prefetch_payload_t<
      T,
      tile_desc_t<matA_load_0_t::tile_size_x, matA_load_0_t::tile_size_y, 1, 1>,
      layout_grad,
      mem_loc_grad,
      1, /*tg_size_x*/
      gpu_arch::Xe>;

  using matA_bpi_t = tile_t<Act_T, matA_tile_desc_t>;

  using matAcc_init_0_t = tile_t<T, matAcc_tile_desc_t0>;

  using prefetch_matAcc_t = prefetch_payload_t<
      T,
      tile_desc_t<
          matAcc_init_0_t::tile_size_x,
          matAcc_init_0_t::tile_size_y,
          1,
          1>,
      layout_grad,
      mem_loc_grad,
      1, /*tg_size_x=1*/
      gpu_arch::Xe>;

  using matC_t0 = tile_t<T, matAcc_tile_desc_t0>;
  using matC_payload_t0 = mem_payload_t<
      T,
      matAcc_tile_desc_t0,
      msg_type_v<matAcc_tile_desc_t0, mem_loc_grad>,
      layout_grad,
      mem_loc_grad,
      gpu_arch::Xe>;

  using matC_t1 = tile_t<T, matAcc_tile_desc_t1>;
  using matC_payload_t1 = mem_payload_t<
      T,
      matAcc_tile_desc_t1,
      msg_type_v<matAcc_tile_desc_t1, mem_loc_grad>,
      layout_grad,
      mem_loc_grad,
      gpu_arch::Xe>;
  static constexpr tdesc_update_dir load_update_config =
      tdesc_update_dir::x_dir;
  using mask_in_t = tile_t<float, matAcc_tile_desc_t1>;
  using mask_payload_t = mem_payload_t<
      float,
      matAcc_tile_desc_t1,
      msg_type_v<matAcc_tile_desc_t1, mem_space::global>,
      mem_layout::row_major,
      mem_space::global,
      gpu_arch::Xe>;

  static void inline call(xetla_exec_item<3>& ei, bpi_config_t<T>* args) {
    matA_tile_desc_t matA_tile_desc;

    mask_in_t mask_in;
    matC_t0 matC_0; /// for hidden grads
    matC_t1 matC_1; /// for layer input grads
    mask_payload_t mask_payload;
    matC_payload_t0 matC_payload_0;
    matC_payload_t1 matC_payload_1;

    matA_load_0_t matBPI_load_reset, matBPI_load_new, matBPI_load_input,
        matBPI_load_g2, matBPI_load_hidden, matBPI_load_init, matBPI_load_layer,
        matBPI_store; /// load data src0
    matA_payload_global_t matBPI_payload_reset, matBPI_payload_new,
        matBPI_payload_input, matBPI_payload_g2, matBPI_payload_hidden,
        matBPI_payload_init, matBPI_payload_layer, matBPI_payload_g;

    matA_bpi_t matBPI, matBPI_0, matBPI_1, matBPI_2; /// dst = cvt<src0 * src1>
    matAcc_init_0_t matAcc_layer, matAcc_init, matAcc_input;
    prefetch_matAcc_t acc_layer, acc_init, acc_input;

    mem_desc_out_t matC_base_desc;
    mem_desc_out_t bpi_desc;
    mem_desc_mask_t mask_desc;

    brgemm_t_0 brgemm_op_0;
    brgemm_t_1 brgemm_op_1;

    brgemm_arguments_hidden brgemm_arg_0;
    brgemm_arguments_input brgemm_arg_1;
    matAcc_t0 matAcc_0;
    matAcc_t1 matAcc_1;
    int gate_id = ei.get_group(0);
    int batch_size, input_size, hidden_size;
    batch_size = args->batch_size;
    input_size = args->input_size;
    hidden_size = args->hidden_size;

    int matrix_n_0, start_x_b_0, start_y_b_0;
    int matrix_n_1, start_x_b_1, start_y_b_1;

    int boundary_n_0, boundary_n_1, boundary_m, boundary_k;
    int wg_tile_k;
    int start_m = ei.get_group(1) * wg_tile_m;
    int start_k = 0;
    int start_n_mask;
    int gate_nums = 3;
    wg_tile_k = hidden_size;
    boundary_k = wg_tile_k;

    boundary_m =
        (start_m + wg_tile_m) > batch_size ? batch_size : (start_m + wg_tile_m);

    int loop_count = (wg_tile_k + sg_tile_k - 1) / sg_tile_k;
    int thread_count = (wg_tile_n_0 + sg_tile_n_0 - 1) / sg_tile_n_0;
    int loop_count_per_thread = (loop_count + thread_count - 1) / thread_count;
    int lhs_start_k = ei.get_local_id(0) * loop_count_per_thread * sg_tile_k;

    int matrix_m = batch_size;
    int matrix_k = hidden_size;
    int start_x_a = start_k;
    int start_y_a = start_m;

    int32_t tile_offset_m = ei.get_local_id(1) * sg_tile_m;
    int offset_x_a = is_col_major_a ? tile_offset_m : 0;
    int offset_y_a = is_col_major_a ? 0 : tile_offset_m;

    int io_size = batch_size * hidden_size;

    int mask_size = batch_size * hidden_size;
    int layer_grad_size = batch_size * input_size;
    int seq_len = args->sequence_length;
    int one_layer_size = io_size * seq_len;

    worker_scope_t g;
    g.init(ei.get_local_linear_id());
    for (unsigned seq_id = 0; seq_id < seq_len; ++seq_id) {
      for (int j = (hidden_size + wg_tile_n_0 - 1) / wg_tile_n_0 - 1; j >= 0;
           --j) {
        int start_n_0 = (ei.get_group(2) + j) * wg_tile_n_0;
        int start_n_1 = (ei.get_group(2) + j) * wg_tile_n_1;

        BPI_CONFIG_SETTING(0, batch_size, hidden_size, hidden_size);
        BPI_CONFIG_SETTING(1, batch_size, hidden_size, input_size);
        MACC_INIT_LOAD(0, args->layer_err_ptr, hidden_size);
        matAcc_0.reg =
            xetla_cvt<Act_T, T, matAcc_t0::tile_elems>(matAcc_init.reg);
        T* init_err_ptr =
            args->partial_err_ptr + (seq_len - 1 - seq_id) * io_size;
        MACC_INIT_LOAD(0, init_err_ptr, hidden_size);
        matAcc_0.reg = matAcc_0.reg +
            xetla_cvt<Act_T, T, matAcc_t0::tile_elems>(matAcc_init.reg);
        SW_BARRIER();
        MACC_INIT_LOAD(
            0,
            args->input_gate_ptr + (seq_len - 1 - seq_id) * io_size,
            hidden_size);
        matAcc_0.reg = matAcc_0.reg *
            xetla_cvt<Act_T, T, matAcc_t0::tile_elems>(matAcc_init.reg);
        matAcc_1.init(0);

        int i = 0;

        BPI_DESC_INIT(0, args->layer_err_ptr);
        matBPI_payload_layer.init(matC_base_desc);
        // matBPI_load_layer.init_tdesc(.get_tdesc());
        prefetch_t layer_prefetch(matC_base_desc);
        layer_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
        // layer_prefetch.template update_prefetch_tdesc<load_update_config>(
        //     sg_tile_k);
        tile_load(matBPI_load_layer, matBPI_payload_layer);
        // matBPI_load_layer
        //     .template load<cache_hint::cached, cache_hint::cached>();
        matBPI_payload_layer.template update_tdesc<load_update_config>(
            sg_tile_k);
        matBPI.reg =
            xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_layer.reg);
        SW_BARRIER();

        BPI_DESC_INIT(0, init_err_ptr);
        matBPI_payload_init.init(matC_base_desc);
        prefetch_t init_prefetch(matC_base_desc);
        init_prefetch.template update_tdesc<load_update_config>(sg_tile_k);

        tile_load(matBPI_load_init, matBPI_payload_init);
        matBPI_payload_init.template update_tdesc<load_update_config>(
            sg_tile_k);
        matBPI.reg = matBPI.reg +
            xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_init.reg);
        SW_BARRIER();

        BPI_DESC_INIT(
            1,
            args->input_gate_ptr +
                (seq_len - 1 - seq_id) * io_size); /// input_gate
        matBPI_payload_input.init(matC_base_desc);
        prefetch_t input_prefetch(matC_base_desc);
        input_prefetch.template update_tdesc<load_update_config>(sg_tile_k);

        tile_load(matBPI_load_input, matBPI_payload_input);
        matBPI_payload_input.template update_tdesc<load_update_config>(
            sg_tile_k);
        matBPI_2.reg =
            (1 -
             xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                 matBPI_load_input.reg));
        matBPI_0.reg = matBPI_2.reg;
        matBPI_1.reg = matBPI_2.reg *
            xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_input.reg);
        SW_BARRIER();

        BPI_DESC_INIT(
            0,
            args->new_gate_ptr + (seq_len - 1 - seq_id) * io_size); /// new_gate
        matBPI_payload_new.init(matC_base_desc);
        prefetch_t new_prefetch(matC_base_desc);
        new_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
        tile_load(matBPI_load_new, matBPI_payload_new);
        matBPI_payload_new.template update_tdesc<load_update_config>(sg_tile_k);
        matBPI_2.reg = matBPI_2.reg *
            (1 -
             xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_new.reg) *
                 xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                     matBPI_load_new.reg));
        matBPI_0.reg = matBPI_2.reg;
        matBPI_2.reg = matBPI_2.reg * matBPI.reg;
        MAT_STORE_GLOBAL(
            2, args->bpi1_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
        SW_BARRIER();

        BPI_DESC_INIT(
            1, args->hidden_ptr + (seq_len - 1 - seq_id) * io_size); // hidden
        matBPI_payload_hidden.init(matC_base_desc);
        prefetch_t hidden_prefetch(matC_base_desc);
        hidden_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
        tile_load(matBPI_load_hidden, matBPI_payload_hidden);
        matBPI_payload_hidden.template update_tdesc<load_update_config>(
            sg_tile_k);
        matBPI_1.reg = matBPI_1.reg *
            (xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                 matBPI_load_hidden.reg) -
             xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_new.reg));
        /// store bpi to global memory
        matBPI_1.reg = matBPI_1.reg * matBPI.reg;
        MAT_STORE_GLOBAL(
            1, args->bpi0_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
        SW_BARRIER();

        BPI_DESC_INIT(
            0,
            args->reset_gate_ptr +
                (seq_len - 1 - seq_id) * io_size); /// reset_gate
        matBPI_payload_reset.init(matC_base_desc);
        prefetch_t reset_prefetch(matC_base_desc);
        reset_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
        tile_load(matBPI_load_reset, matBPI_payload_reset);
        matBPI_payload_reset.template update_tdesc<load_update_config>(
            sg_tile_k);
        MAT_STORE_GLOBAL(
            1, args->bpi1_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
        matBPI_2.reg = matBPI_2.reg *
            xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_reset.reg);
        MAT_STORE_GLOBAL(
            2, args->bpi0_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
        matBPI_0.reg = matBPI_2.reg *
            (1 -
             xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                 matBPI_load_reset.reg));
        /// store bpi to global memory
        SW_BARRIER();

        BPI_DESC_INIT(
            1, args->hgate_2_ptr + (seq_len - 1 - seq_id) * io_size); // h2_gate
        matBPI_payload_g2.init(matC_base_desc);
        prefetch_t g2_prefetch(matC_base_desc);
        g2_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
        tile_load(matBPI_load_g2, matBPI_payload_g2);
        matBPI_payload_g2.template update_tdesc<load_update_config>(sg_tile_k);
        matBPI_0.reg = matBPI_0.reg *
            xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_g2.reg);
        /// store bpi to global memory
        MAT_STORE_GLOBAL(
            0, args->bpi0_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
        MAT_STORE_GLOBAL(
            0, args->bpi1_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);

#pragma unroll
        for (i = 1; i < prefetch_distance; i++) {
          tile_prefetch(layer_prefetch);
          layer_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          tile_prefetch(init_prefetch);
          init_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          tile_prefetch(input_prefetch);
          input_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          tile_prefetch(new_prefetch);
          new_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          tile_prefetch(hidden_prefetch);
          hidden_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          tile_prefetch(reset_prefetch);
          reset_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          tile_prefetch(g2_prefetch);
          g2_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
        }
        for (int i = 1; lhs_start_k + i * sg_tile_k < hidden_size; i++) {
          tile_load(matBPI_load_layer, matBPI_payload_layer);
          matBPI_payload_layer.template update_tdesc<load_update_config>(
              sg_tile_k);
          tile_prefetch(layer_prefetch);
          layer_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          matBPI.reg = xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
              matBPI_load_layer.reg);
          SW_BARRIER();

          tile_load(matBPI_load_init, matBPI_payload_init);
          matBPI_payload_init.template update_tdesc<load_update_config>(
              sg_tile_k);
          tile_prefetch(init_prefetch);
          init_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          matBPI.reg = matBPI.reg +
              xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_init.reg);
          SW_BARRIER();

          tile_load(matBPI_load_input, matBPI_payload_input);
          matBPI_payload_input.template update_tdesc<load_update_config>(
              sg_tile_k);
          tile_prefetch(input_prefetch);
          input_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          matBPI_2.reg =
              (1 -
               xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                   matBPI_load_input.reg));
          matBPI_0.reg = matBPI_2.reg;
          matBPI_1.reg = matBPI_2.reg *
              xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                             matBPI_load_input.reg);

          SW_BARRIER();
          tile_load(matBPI_load_new, matBPI_payload_new);
          matBPI_payload_new.template update_tdesc<load_update_config>(
              sg_tile_k);
          tile_prefetch(new_prefetch);
          new_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          matBPI_2.reg = matBPI_2.reg *
              (1 -
               xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                   matBPI_load_new.reg) *
                   xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                       matBPI_load_new.reg));
          matBPI_0.reg = matBPI_2.reg;

          matBPI_2.reg = matBPI_2.reg * matBPI.reg;
          MAT_STORE_GLOBAL(
              2, args->bpi1_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
          SW_BARRIER();
          tile_load(matBPI_load_hidden, matBPI_payload_hidden);
          matBPI_payload_hidden.template update_tdesc<load_update_config>(
              sg_tile_k);
          tile_prefetch(hidden_prefetch);
          hidden_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          matBPI_1.reg = matBPI_1.reg *
              (xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                   matBPI_load_hidden.reg) -
               xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                   matBPI_load_new.reg));
          /// store bpi to global memory
          matBPI_1.reg = matBPI_1.reg * matBPI.reg;
          MAT_STORE_GLOBAL(
              1, args->bpi0_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);

          SW_BARRIER();
          tile_load(matBPI_load_reset, matBPI_payload_reset);
          matBPI_payload_reset.template update_tdesc<load_update_config>(
              sg_tile_k);
          tile_prefetch(reset_prefetch);
          reset_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          MAT_STORE_GLOBAL(
              1, args->bpi1_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
          matBPI_2.reg = matBPI_2.reg *
              xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                             matBPI_load_reset.reg);
          MAT_STORE_GLOBAL(
              2, args->bpi0_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
          matBPI_0.reg = matBPI_2.reg *
              (1 -
               xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(
                   matBPI_load_reset.reg));
          /// store bpi to global memory
          SW_BARRIER();
          tile_load(matBPI_load_g2, matBPI_payload_g2);
          matBPI_payload_g2.template update_tdesc<load_update_config>(
              sg_tile_k);
          tile_prefetch(g2_prefetch);
          g2_prefetch.template update_tdesc<load_update_config>(sg_tile_k);
          matBPI_0.reg = matBPI_0.reg *
              xetla_cvt<Act_T, T, matA_bpi_t::tile_elems>(matBPI_load_g2.reg);
          /// store bpi to global memory
          MAT_STORE_GLOBAL(
              0, args->bpi0_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
          MAT_STORE_GLOBAL(
              0, args->bpi1_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
        }
        SW_BARRIER();
        __esimd_barrier();
        /// [h_0, h_1, h_2] x [w_hr, w_hz, w_hn]^T
        BPI_BRGEMM_CALL(
            0,
            args->w_h_ptr,
            args->bpi0_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);

        BPI_MATC_STORE_GLOBAL(0, args->x0_grad_ptr, hidden_size);
        SW_BARRIER();
        __esimd_barrier();
        BPI_BRGEMM_CALL(
            1,
            args->w_i_ptr,
            args->bpi1_ptr + (seq_len - 1 - seq_id) * io_size * gate_nums);
        if (args->dropout > 0) {
          DROPOUT(1, args->mask_ptr + (seq_len - 1 - seq_id) * mask_size);
        }
        /// [g_0, g_1, g_2] x [w_ir, w_iz, w_in]^T
        BPI_MATC_STORE_GLOBAL(
            1,
            args->x_grad_ptr + (seq_len - 1 - seq_id) * layer_grad_size,
            input_size);
        SW_BARRIER();
      }
      args->layer_err_ptr = args->x0_grad_ptr;
    }
  }
};

template <
    typename input_T,
    typename Act_T,
    uint32_t wg_tile_m_t,
    uint32_t wg_tile_n_0_t,
    uint32_t wg_tile_n_1_t,
    uint32_t sg_tile_m_t,
    uint32_t sg_tile_n_0_t,
    uint32_t sg_tile_n_1_t,
    uint32_t sg_tile_k_t>
struct kernel_xcoder_gru_bpi {
  /// @brief
  /// @param ei
  /// @param layer_err_ptr    err inputs  from last cell per layer shape =
  /// layer_size x batch_size x hidden_size
  /// @param y_err_ptr  err inputs from last layer shape = sequence x batch_size
  /// x hidden_size
  /// @param x_grad_ptr       grad outputs for first layer inputs   shape =
  /// sequence x batch_size x input_size
  /// @param partial_grad_ptr partial grad inputs layer grads   shape = sequence
  /// x batch_size x hidden_size
  /// @param x0_grad_ptr      err output for fisrt cell per layer  shape =
  /// layer_size x batch_size x hidden_size
  /// @param reset_gate_ptr
  /// @param input_gate_ptr
  /// @param new_gate_ptr
  /// @param hgate_2_ptr
  /// @param i_weights
  /// @param h_weights
  /// @param batch_size
  /// @param input_size
  /// @param hidden_size
  /// @param sequence_length
  /// @param layer_size
  /// @param dropout
  static void inline run(
      xetla_exec_item<3> ei,
      input_T* layer_err_ptr,
      input_T* y_err_ptr,
      input_T* x_grad_ptr,
      input_T* bpi0_ptr,
      input_T* bpi1_ptr,
      input_T* partial_grad_ptr,
      input_T* x0_grad_ptr,
      input_T* reset_gate_ptr,
      input_T* input_gate_ptr,
      input_T* new_gate_ptr,
      input_T* hgate_2_ptr,
      input_T* hidden_ptr,
      input_T* i_weights,
      input_T* h_weights,
      float* mask_ptr,
      int batch_size,
      int input_size,
      int hidden_size,
      int sequence_length,
      int layer_size,
      float dropout = 0) {
    constexpr uint32_t fused_op_wg_m = wg_tile_m_t;
    constexpr uint32_t fused_op_wg_n_0 = wg_tile_n_0_t;
    constexpr uint32_t fused_op_wg_n_1 = wg_tile_n_1_t;
    constexpr uint32_t fused_op_sg_m = sg_tile_m_t;
    constexpr uint32_t fused_op_sg_n_0 = sg_tile_n_0_t;
    constexpr uint32_t fused_op_sg_n_1 = sg_tile_n_1_t;
    constexpr uint32_t fused_op_sg_k = sg_tile_k_t;
    using fused_op_0 = gru_layer_bpi<
        input_T,
        Act_T,
        fused_op_wg_m,
        fused_op_wg_n_0,
        fused_op_wg_n_1,
        fused_op_sg_m,
        fused_op_sg_n_0,
        fused_op_sg_n_1,
        fused_op_sg_k>;
    using fused_op_1 = gru_layer_bpi<
        input_T,
        Act_T,
        fused_op_wg_m,
        fused_op_wg_n_0,
        fused_op_wg_n_0,
        fused_op_sg_m,
        fused_op_sg_n_0,
        fused_op_sg_n_0,
        fused_op_sg_k>;
    bpi_config_t<input_T> args;
    int layer_input_size = batch_size * input_size;
    int hidden_io_size = batch_size * hidden_size;
    int input_weight_size = input_size * hidden_size;
    int hidden_weight_size = hidden_size * hidden_size;
    int one_layer_size = hidden_io_size * sequence_length;
    int one_layer_io_size = hidden_io_size * (sequence_length + 1);
    int gate_nums = 3;

    args.sequence_length = sequence_length;
    args.batch_size = batch_size;
    args.hidden_size = hidden_size;
    args.input_size = hidden_size;
    args.reserve = 1.0f / (1.0f - dropout);
    args.dropout = dropout;
    int ping = 0;
    int pong = 1;
    for (unsigned layer_id = layer_size - 1; layer_id > 0; --layer_id) {
      args.slm_addr = 0;
      args.layer_err_ptr = layer_err_ptr + layer_id * hidden_io_size;
      args.partial_err_ptr = layer_id == layer_size - 1
          ? y_err_ptr
          : partial_grad_ptr + ping * one_layer_size;
      args.x_grad_ptr = partial_grad_ptr + pong * one_layer_size;
      args.bpi0_ptr = bpi0_ptr + layer_id * one_layer_size * gate_nums;
      args.bpi1_ptr = bpi1_ptr + layer_id * one_layer_size * gate_nums;
      args.x0_grad_ptr = x0_grad_ptr + layer_id * hidden_io_size;
      args.hidden_ptr = hidden_ptr + layer_id * one_layer_io_size;
      args.reset_gate_ptr = reset_gate_ptr + layer_id * one_layer_size;
      args.input_gate_ptr = input_gate_ptr + layer_id * one_layer_size;
      args.new_gate_ptr = new_gate_ptr + layer_id * one_layer_size;
      args.hgate_2_ptr = hgate_2_ptr + layer_id * one_layer_size;
      args.w_i_ptr = i_weights + 3 * input_weight_size +
          (layer_id - 1) * hidden_weight_size * 3;
      args.w_h_ptr = h_weights + 3 * hidden_weight_size * layer_id;
      args.mask_ptr = mask_ptr + (layer_id - 1) * one_layer_size;
      SW_BARRIER();
      fused_op_1::call(ei, &args);
      ping = (ping + 1) % 2;
      pong = (pong + 1) % 2;
    }
    args.slm_addr = 0;
    args.layer_err_ptr = layer_err_ptr;
    args.partial_err_ptr =
        layer_size == 1 ? y_err_ptr : partial_grad_ptr + ping * one_layer_size;
    args.x_grad_ptr = x_grad_ptr;
    args.bpi0_ptr = bpi0_ptr;
    args.bpi1_ptr = bpi1_ptr;
    args.x0_grad_ptr = x0_grad_ptr;
    args.hidden_ptr = hidden_ptr;
    args.reset_gate_ptr = reset_gate_ptr;
    args.input_gate_ptr = input_gate_ptr;
    args.new_gate_ptr = new_gate_ptr;
    args.hgate_2_ptr = hgate_2_ptr;
    args.w_i_ptr = i_weights;
    args.w_h_ptr = h_weights;
    args.batch_size = batch_size;
    args.hidden_size = hidden_size;
    args.input_size = input_size;
    args.reserve = 0;
    args.dropout = 0;
    SW_BARRIER();
    fused_op_0::call(ei, &args);
  }
};

// extern "C"
template <typename gru_bpi_config_t>
void gru_backward_data_impl(
    void* layer_err_ptr,
    void* y_err_ptr,
    void* x_grad_ptr,
    void* bpi0_ptr,
    void* bpi1_ptr,
    void* partial_grad_ptr,
    void* x0_grad_ptr,
    void* reset_gate_ptr,
    void* input_gate_ptr,
    void* new_gate_ptr,
    void* hgate_2_ptr,
    void* hidden_ptr,
    void* i_weights,
    void* h_weights,
    void* mask_ptr,
    int batch_size,
    int input_size,
    int hidden_size,
    int sequence_length,
    int layer_size,
    float dropout,
    cl::sycl::queue& Queue) {
  static size_t wg_tile_m = gru_bpi_config_t::wg_tile_m;
  static size_t wg_tile_n_0 = gru_bpi_config_t::wg_tile_n_0;
  static size_t wg_tile_n_1 = gru_bpi_config_t::wg_tile_n_1;
  static size_t sg_tile_m = gru_bpi_config_t::sg_tile_m;
  static size_t sg_tile_n_0 = gru_bpi_config_t::sg_tile_n_0;
  static size_t sg_tile_n_1 = gru_bpi_config_t::sg_tile_n_1;
  static size_t sg_tile_k = gru_bpi_config_t::sg_tile_k;

  using input = gru_bpi_config_t::input_T;
  using Act = gru_bpi_config_t::Act_T;

  size_t N = batch_size;
  size_t H = hidden_size;
  cl::sycl::range<3> GroupRange{
      1,
      (N + wg_tile_m - 1) / wg_tile_m,
      // (H + wg_tile_n_0 - 1) / wg_tile_n_0
      1};
  cl::sycl::range<3> LocalRange{
      1,
      (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
      (wg_tile_n_0 + sg_tile_n_0 - 1) / sg_tile_n_0};
  cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(Range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item ei(item);
      using xcoder_gru_bpi_op = kernel_xcoder_gru_bpi<
          typename gru_bpi_config_t::input_T,
          typename gru_bpi_config_t::Act_T,
          gru_bpi_config_t::wg_tile_m,
          gru_bpi_config_t::wg_tile_n_0,
          gru_bpi_config_t::wg_tile_n_1,
          gru_bpi_config_t::sg_tile_m,
          gru_bpi_config_t::sg_tile_n_0,
          gru_bpi_config_t::sg_tile_n_1,
          gru_bpi_config_t::sg_tile_k>;
      xcoder_gru_bpi_op::run(
          ei,
          (input*)layer_err_ptr,
          (input*)y_err_ptr, /* inputs*/
          (input*)x_grad_ptr,
          (input*)bpi0_ptr,
          (input*)bpi1_ptr,
          (input*)partial_grad_ptr,
          (input*)x0_grad_ptr, /* outputs*/
          (input*)reset_gate_ptr,
          (input*)input_gate_ptr,
          (input*)new_gate_ptr,
          (input*)hgate_2_ptr,
          (input*)hidden_ptr, /*workspace*/
          (input*)i_weights,
          (input*)h_weights, /*weights*/
          (float*)mask_ptr,
          batch_size,
          input_size,
          hidden_size,
          sequence_length,
          layer_size,
          dropout);
    });
  };
  DPCPP_Q_SUBMIT(Queue, cgf);
}

} // namespace xetla
} // namespace xpu
