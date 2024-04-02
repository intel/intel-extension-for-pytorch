#include <utils/DPCPP.h>
#include <limits>
#include "../../../comm/AccumulateType.h"
#include "../../mha.h"
#include "fmha_backward_policy.h"
#include "fmha_forward_policy.h"
#include "fmha_utils.h"

namespace gpu::xetla {

namespace fmha {

template <
    typename fmha_policy,
    typename scalar_t,
    bool kUseBias,
    bool kIsCausal,
    bool kIsDropout>
class fmha_backward_t {
 public:
  using accum_t = float;
  static constexpr accum_t kNegInfinity = INFINITY * -1;

  struct arguments_t {
    // Input tensors
    scalar_t* dO_ptr; // [B, F, N, H] - grad_output
    scalar_t* Q_ptr; // [B, F, N, H] -> query
    scalar_t* K_ptr; // [B, T, N, H] -> key
    scalar_t* V_ptr; // [B, T, N, H] -> value
    scalar_t* B_ptr = nullptr; // [B, 1/N, F, T] - bias
    uint8_t* Dp_ptr = nullptr; // [B, N, F, T] - dropout mask
    scalar_t* O_ptr; // [B, F, N, H] - output
    accum_t* L_ptr; // [B, N, F]
    accum_t* D_ptr; // [B, N, F]
    accum_t* dQ_acc_ptr;
    // Dropout scale is computed from dropout prob
    accum_t dp_prob;
    accum_t dp_scale;
    // Output tensor
    scalar_t* dQ_ptr;
    scalar_t* dK_ptr;
    scalar_t* dV_ptr;
    scalar_t* dB_ptr = nullptr;
    // Dimension size
    uint32_t uB;
    uint32_t uN;
    uint32_t uH;
    uint32_t uF;
    uint32_t uT;

    uint32_t bias_strideB;
    uint32_t bias_strideN;
    uint32_t bias_strideF;
    uint32_t uMT;
    // Softmax scale is the reciprocal square root of head size by default
    accum_t sm_scale;
    // seed/offset used to generate dropout mask
    uint64_t seed;
    uint64_t offset;
    bool is_bias_add;

    inline arguments_t() = default;
    inline arguments_t(
        scalar_t* grad_out,
        scalar_t* query,
        scalar_t* key,
        scalar_t* value,
        scalar_t* bias,
        uint8_t* dropout,
        scalar_t* out,
        accum_t* l,
        accum_t* d,
        accum_t* grad_q_tmp,
        accum_t dropout_prob,
        scalar_t* grad_query,
        scalar_t* grad_key,
        scalar_t* grad_value,
        scalar_t* grad_bias,
        uint32_t num_batches,
        uint32_t num_heads,
        uint32_t head_size,
        uint32_t num_queries,
        uint32_t num_keys,
        uint32_t bias_strideB,
        uint32_t bias_strideN,
        uint32_t bias_strideF,
        uint32_t attn_mask_padded_block_size,
        accum_t sm_scale,
        uint64_t seed,
        uint64_t offset)
        : dO_ptr(grad_out),
          Q_ptr(query),
          K_ptr(key),
          V_ptr(value),
          B_ptr(bias),
          Dp_ptr(dropout),
          O_ptr(out),
          L_ptr(l),
          D_ptr(d),
          dQ_acc_ptr(grad_q_tmp),
          dp_prob(dropout_prob),
          dp_scale(1.f / (1.f - dropout_prob)),
          dQ_ptr(grad_query),
          dK_ptr(grad_key),
          dV_ptr(grad_value),
          dB_ptr(grad_bias),
          uB(num_batches),
          uN(num_heads),
          uH(head_size),
          uF(num_queries),
          uT(num_keys),
          bias_strideB(bias_strideB),
          bias_strideN(bias_strideN),
          bias_strideF(bias_strideF),
          uMT(attn_mask_padded_block_size),
          sm_scale(sm_scale),
          seed(seed),
          offset(offset),
          is_bias_add(bias_strideF == 0) {}
  };

 private:
  // -------------------- // Compute policy // -------------------- //
  static constexpr uint32_t accum_step = fmha_policy::accum_step;
  static constexpr uint32_t stages = fmha_policy::stages;
  static constexpr uint32_t sync_freq = fmha_policy::sync_freq;

  using compute_attr = group::compute_attr_t<scalar_t, scalar_t, accum_t>;
  using perf_tuning_knob =
      group::perf_tuning_knob_t<accum_step, stages, sync_freq>;
  using compute_policy = group::
      compute_policy_default_xmx<compute_attr, perf_tuning_knob, gpu_arch::Xe>;

  // ---------------- // Tile shape and Threads // ---------------- //
  static constexpr uint32_t kBr = fmha_policy::kBr;
  static constexpr uint32_t kBc = fmha_policy::kBc;
  static constexpr uint32_t kHm = fmha_policy::kHm;
  static constexpr uint32_t kSgBr = fmha_policy::kSgBr;
  static constexpr uint32_t kSgBc = fmha_policy::kSgBc;
  static constexpr uint32_t kSgHm = fmha_policy::kSgHm;
  static constexpr uint32_t kBcHm_SgBc = fmha_policy::kBcHm_SgBc;

  using tile_shape_BrBc = group::tile_shape_t<kBc, kBr, kSgBc, kSgBr>;
  using tile_shape_BrHm = group::tile_shape_t<kHm, kBr, kSgHm, kSgBr>;
  using tile_shape_BcHm = group::tile_shape_t<kHm, kBc, kSgHm, kBcHm_SgBc>;

  static constexpr uint32_t wg_size_x = tile_shape_BrBc::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_BrBc::wg_size_y;
  using work_group_BrBc_t = typename tile_shape_BrBc::work_group_t;
  using work_group_BrHm_t = typename tile_shape_BrHm::work_group_t;
  using work_group_BcHm_t = typename tile_shape_BcHm::work_group_t;
  static constexpr uint32_t wg_size = work_group_BrBc_t::size;

  static_assert(
      kHm / kSgHm == kBc / kSgBc,
      "wg_size_x must be the same between Hm and Bc");
  static_assert(
      kBr / kSgBr == kBc / kBcHm_SgBc,
      "wg_size_y must be the same between Br and Bc_M");
  static_assert(wg_size <= 32, "The number of threads should be less than 32!");

  // --------------------- // Memory desc // ---------------------- //
  // suffix: L -> local; T -> transpose
  using mem_desc_dOi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Qi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Kj_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Kj_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_Vj_T_t =
      mem_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_Oi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Bij_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Li_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Di_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Pij_L_T_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_dSij_L_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_dSij_L_T_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_dQi_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dQi_acc_t =
      mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dKj_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dVj_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_dBij_t =
      mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Dp_Mask_t =
      mem_desc_t<uint8_t, mem_layout::row_major, mem_space::global>;

  // ------------------- // Slm and nbarrier // ------------------- //
  // Todo: consider parallel Q(Br) as fwd did
  static constexpr uint32_t slm_size_Pij = kBr * kBc * sizeof(scalar_t);
  static constexpr uint32_t slm_size_Sij = kBr * kBc * sizeof(scalar_t);
  static constexpr uint32_t slm_size_D =
      (wg_size_x > 1) ? wg_size * kSgBr * sizeof(accum_t) : 0;
  // Slm addr to store inermediate results
  static constexpr uint32_t Pij_slm = /* slm_base */ 0;
  static constexpr uint32_t Sij_slm = Pij_slm + slm_size_Pij;
  static constexpr uint32_t D_slm = Sij_slm + slm_size_Sij;

  static constexpr uint32_t nbarrier_cnt = wg_size_x + wg_size_y;
  // Define kernel to compute Sij = Qi x Kj.T
  using gemm_Sij_t = group::
      gemm_t<compute_policy, tile_shape_BrBc, mem_desc_Qi_t, mem_desc_Kj_T_t>;
  using matAcc_Sij_t = typename gemm_Sij_t::matAcc_t;
  using dropout_fd_t = dropout_fwd_t<matAcc_Sij_t::tile_elems>;
  using dp_mask_tile_desc_t = typename gemm_Sij_t::matAcc_tile_desc_t;
  using dp_mask_tile_t = subgroup::tile_t<uint8_t, dp_mask_tile_desc_t>;
  // ======================== // Context // ======================= //

  /// @brief Used to store variables in the flash mha loops
  struct context_t {
    // thread id
    work_group_BrBc_t g_brbc;
    work_group_BrHm_t g_brhm;
    work_group_BcHm_t g_bchm;
    uint32_t sg_idx;
    uint32_t sg_idy;
    // nbarrier
    xetla_nbarrier_t<wg_size_x, wg_size_x> nbarrier;
    xetla_nbarrier_t<wg_size_y, wg_size_y> nbarrier_y;
    // softmax statistics

    // mem desc variables
    mem_desc_dOi_t mem_desc_dOi;
    mem_desc_Qi_t mem_desc_Qi;
    mem_desc_Kj_t mem_desc_Kj;
    mem_desc_Kj_T_t mem_desc_Kj_T;
    mem_desc_Vj_T_t mem_desc_Vj_T;
    mem_desc_Oi_t mem_desc_Oi;
    mem_desc_Bij_t mem_desc_Bij;
    mem_desc_Li_t mem_desc_Li;
    mem_desc_Di_t mem_desc_Di;
    mem_desc_Pij_L_T_t mem_desc_Pij_L_T;
    mem_desc_dSij_L_t mem_desc_dSij_L;
    mem_desc_dSij_L_T_t mem_desc_dSij_L_T;
    mem_desc_dQi_t mem_desc_dQi;
    mem_desc_dQi_acc_t mem_desc_dQi_acc;
    mem_desc_dKj_t mem_desc_dKj;
    mem_desc_dVj_t mem_desc_dVj;
    mem_desc_dBij_t mem_desc_dBij;
    mem_desc_Dp_Mask_t mem_desc_Dpij;
    dropout_fd_t dropout_op;

    inline context_t() = default;

    /// @brief Initialize invariant variables in the flash mha loop
    inline void init_context(const nd_item<3>& item, const arguments_t& args) {
      // thread id
      uint32_t sg_id = item.get_local_linear_id();
      g_brbc.init(sg_id);
      g_brhm.init(sg_id);
      g_bchm.init(sg_id);
      sg_idx = sg_id % wg_size_x;
      sg_idy = sg_id / wg_size_x;
      // nbarrier
      nbarrier.init_nbarrier(sg_idy, nbarrier_role::producer_consumer);
      nbarrier_y.init_nbarrier(
          wg_size_y + sg_idx, nbarrier_role::producer_consumer);
      // softmax statistics
    }

    /// @brief Initialize update for dQ variables in the flash mha loop
    inline void update_context_dQ(
        const nd_item<3>& item,
        const arguments_t& args,
        uint32_t startF) {
      // mem desc variables
      uint32_t gid = item.get_group(0);
      uint32_t bid = item.get_group(0) / args.uN;
      uint32_t nid = item.get_group(0) % args.uN;
      int32_t start_x = nid * args.uH;
      uint32_t end_x = start_x + args.uH;
      int32_t start_y = bid * args.uF + startF;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = (bid + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;
      uint32_t pitch = args.uH * args.uN;

      mem_desc_dQi.init(args.dQ_ptr, {end_x, end_y, pitch}, {start_x, start_y});
      mem_desc_dQi_acc.init(
          args.dQ_acc_ptr,
          {end_x, end_y, pitch},
          {int32_t(start_x + sg_idx * kSgHm),
           int32_t(start_y + sg_idy * kSgBr)});
    }

    /// @brief Initialize update for D variables in the flash mha loop
    inline void update_context_D(
        const nd_item<3>& item,
        const arguments_t& args,
        uint32_t startF) {
      // mem desc variables
      uint32_t gid = item.get_group(0);
      uint32_t bid = item.get_group(0) / args.uN;
      uint32_t nid = item.get_group(0) % args.uN;
      int32_t start_x = nid * args.uH;
      uint32_t end_x = start_x + args.uH;
      int32_t start_y = bid * args.uF + startF;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = (bid + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;
      uint32_t pitch = args.uH * args.uN;

      mem_desc_Oi.init(
          args.O_ptr,
          {end_x, end_y, pitch},
          {int32_t(start_x + sg_idx * kSgHm),
           int32_t(start_y + sg_idy * kSgBr)});
      mem_desc_dOi.init(
          args.dO_ptr,
          {end_x, end_y, pitch},
          {int32_t(start_x + sg_idx * kSgHm),
           int32_t(start_y + sg_idy * kSgBr)});

      int32_t start_x_ml = startF + sg_idy * kSgBr;
      int32_t start_y_ml = gid;
      mem_desc_Di.init(
          args.D_ptr,
          {args.uF, args.uB * args.uN, args.uF},
          {start_x_ml, start_y_ml});
    }

    /// @brief Initialize update for Q variables in the flash mha loop
    inline void update_context_Q(
        const nd_item<3>& item,
        const arguments_t& args,
        uint32_t startF,
        uint32_t startT) {
      // mem desc variables
      uint32_t gid = item.get_group(0);
      uint32_t bid = item.get_group(0) / args.uN;
      uint32_t nid = item.get_group(0) % args.uN;
      int32_t start_x = nid * args.uH;
      uint32_t end_x = start_x + args.uH;
      int32_t start_y = bid * args.uF + startF;
      uint32_t end_y = start_y + kBr;
      uint32_t boundary_y = (bid + 1) * args.uF;
      end_y = end_y > boundary_y ? boundary_y : end_y;
      uint32_t pitch = args.uH * args.uN;

      mem_desc_Qi.init(args.Q_ptr, {end_x, end_y, pitch}, {start_x, start_y});
      mem_desc_dQi_acc.init(
          args.dQ_acc_ptr,
          {end_x, end_y, pitch},
          {int32_t(start_x + sg_idx * kSgHm),
           int32_t(start_y + sg_idy * kSgBr)});
      mem_desc_dOi.init(args.dO_ptr, {end_x, end_y, pitch}, {start_x, start_y});

      int32_t start_x_ml = startF + sg_idy * kSgBr;
      int32_t start_y_ml = gid;
      mem_desc_Li.init(
          args.L_ptr,
          {args.uF, args.uB * args.uN, args.uF},
          {start_x_ml, start_y_ml});
      mem_desc_Di.init(
          args.D_ptr,
          {args.uF, args.uB * args.uN, args.uF},
          {start_x_ml, start_y_ml});

      mem_desc_Pij_L_T.init(Pij_slm, {kBr, kBc, kBr}, {0, 0});
      mem_desc_dSij_L.init(Sij_slm, {kBc, kBr, kBc}, {0, 0});
      mem_desc_dSij_L_T.init(Sij_slm, {kBr, kBc, kBr}, {0, 0});

      if constexpr (kIsDropout) {
        if (args.Dp_ptr) {
          int32_t start_x = startT;
          uint32_t end_x = args.uT;
          int32_t start_y = gid * args.uF + startF;
          uint32_t end_y = start_y + kBr;
          uint32_t boundary_y = (gid + 1) * args.uF;
          end_y = end_y > boundary_y ? boundary_y : end_y;

          mem_desc_Dpij.init(
              args.Dp_ptr, {end_x, end_y, args.uT}, {start_x, start_y});
          int32_t tile_offset_x = sg_idx * kSgBc;
          int32_t tile_offset_y = sg_idy * kSgBr;
          mem_desc_Dpij.update_coord(tile_offset_x, tile_offset_y);
        } else {
          int coord_y = startF + sg_idy * kSgBr;
          int coord_x = startT + sg_idx * kSgBc;
          uint64_t sg_subseq = uint64_t(coord_y) << 32 | uint64_t(coord_x);
          uint32_t threshold = uint32_t(args.dp_prob * float(4294967296));
          dropout_op.init(
              args.seed, sg_subseq, args.offset, threshold, args.dp_scale);
        }
      }

      if constexpr (kUseBias) {
        start_x = startT;
        end_x = start_x + kBc;
        uint32_t boundary_x = args.uMT;
        end_x = end_x > boundary_x ? boundary_x : end_x;

        int32_t offset =
            (bid * args.bias_strideB + nid * args.bias_strideN) / args.uMT;
        int32_t start_y = offset + (args.is_bias_add ? 0 : startF);
        uint32_t end_y = args.is_bias_add ? start_y : start_y + kBr;
        uint32_t boundary_y = args.is_bias_add ? start_y : offset + args.uF;
        end_y = end_y > boundary_y ? boundary_y : end_y;

        mem_desc_Bij.init(
            args.B_ptr, {end_x, end_y, args.uMT}, {start_x, start_y});

        start_y = gid * args.uF + startF;
        end_y = start_y + kBr;
        boundary_y = (gid + 1) * args.uF;
        end_y = end_y > boundary_y ? boundary_y : end_y;
        mem_desc_dBij.init(
            args.dB_ptr, {end_x, end_y, args.uMT}, {start_x, start_y});
      }
    }

    /// @brief Update variables KV for each flash mha loop
    inline void update_context_KV(
        const nd_item<3>& item,
        const arguments_t& args,
        uint32_t startT) {
      uint32_t gid = item.get_group(0);
      uint32_t bid = gid / args.uN;
      uint32_t nid = gid % args.uN;

      int32_t start_x = bid * args.uT + startT;
      uint32_t end_x = start_x + kBc;
      uint32_t boundary_x = (bid + 1) * args.uT;
      end_x = end_x > boundary_x ? boundary_x : end_x;
      int32_t start_y = nid * args.uH;
      uint32_t end_y = start_y + args.uH;

      uint32_t pitch = args.uN * args.uH;

      mem_desc_Kj.init(args.K_ptr, {end_y, end_x, pitch}, {start_y, start_x});
      mem_desc_Kj_T.init(args.K_ptr, {end_x, end_y, pitch}, {start_x, start_y});
      mem_desc_Vj_T.init(args.V_ptr, {end_x, end_y, pitch}, {start_x, start_y});

      mem_desc_dKj.init(args.dK_ptr, {end_y, end_x, pitch}, {start_y, start_x});
      mem_desc_dVj.init(args.dV_ptr, {end_y, end_x, pitch}, {start_y, start_x});
    }
  };

  context_t ctx;

  // ======================= // gemm_Sij // ======================= //

  /// @brief gemm_Sij is used to compute Sij = Qi x Kj.T
  /// # [Br,H] x [H,Bc] = [Br,Bc]
  inline void gemm_Sij(matAcc_Sij_t* matAcc_Sij, const arguments_t& args) {
    using gemm_args_t = typename gemm_Sij_t::arguments_t;

    // Gemm to comput Sij
    gemm_Sij_t gemm;
    uint32_t loop_count = (args.uH + accum_step - 1) / accum_step;
    gemm_args_t gemm_args(ctx.mem_desc_Qi, ctx.mem_desc_Kj_T, loop_count);
    gemm(
        ctx.g_brbc,
        *matAcc_Sij,
        gemm_args,
        0,
        /* nbarrier_base */ nbarrier_cnt);

    // Multiply by softmax scaling factor
    matAcc_Sij->reg *= args.sm_scale;

    // Add bias if needed
    if constexpr (kUseBias) {
      if (args.is_bias_add) {
        using mask_op_t = bias_add_op_t<scalar_t, gpu_arch::Xe>;
        using mask_args_t = typename mask_op_t::arguments_t;
        int32_t tile_offset_x = ctx.sg_idx * kSgBc;
        ctx.mem_desc_Bij.update_coord_x(tile_offset_x);
        mask_op_t mask_op;
        mask_args_t mask_args(ctx.mem_desc_Bij.base, ctx.mem_desc_Bij.shape);
        mask_op(*matAcc_Sij, ctx.mem_desc_Bij.coord, mask_args);
      } else {
        using bias_op_t = subgroup::
            elemwise_reduce_op_t<reduce_op::sum, scalar_t, gpu_arch::Xe>;
        using bias_args_t = typename bias_op_t::arguments_t;

        int32_t tile_offset_x = ctx.sg_idx * kSgBc;
        int32_t tile_offset_y = ctx.sg_idy * kSgBr;
        ctx.mem_desc_Bij.update_coord(tile_offset_x, tile_offset_y);

        bias_op_t bias_op;
        bias_args_t bias_args(ctx.mem_desc_Bij.base, ctx.mem_desc_Bij.shape);
        bias_op(*matAcc_Sij, ctx.mem_desc_Bij.coord, bias_args);
      }
    }
  }

  // ====================== // apply_mask // ====================== //

  /// @brief apply mask to matAccSij.
  inline void apply_mask(
      matAcc_Sij_t* matAcc_Sij,
      const arguments_t& args,
      uint32_t startF,
      uint32_t startT) {
    using tile_mask = tile_mask_t<matAcc_Sij_t>;

    uint32_t sg_startT = startT + ctx.sg_idx * kSgBc;
    uint32_t remainT = std::max(int(args.uT) - int(sg_startT), 0);
    if (remainT < kSgBc) {
      tile_mask::padding_mask(*matAcc_Sij, remainT);
    }

    if constexpr (kIsCausal) {
      uint32_t sg_startF = startF + ctx.sg_idy * kSgBr;
      if (sg_startT + kSgBc > sg_startF) {
        tile_mask::causal_mask(*matAcc_Sij, sg_startT, sg_startF);
      }
    }
  }

  // ====================== // softmax_fwd // ===================== //

  /// @brief softmax_fwd is used to do softmax, Pij = softmax(Sij)
  inline void softmax_fwd(
      matAcc_Sij_t* matAcc_Sij,
      matAcc_Sij_t* matAcc_Sij_drop,
      dp_mask_tile_t* mask_in,
      const arguments_t& args) {
    using load_desc =
        subgroup::tile_desc_t<kSgBr, 1, kSgBr, 1, reg_layout::tiled>;
    using load_tile_t = subgroup::tile_t<accum_t, load_desc>;
    using load_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
        load_desc,
        msg_type::block_2d,
        gpu_arch::Xe>;

    load_tile_t matL_load;
    load_payload_t load_payload(ctx.mem_desc_Li);
    subgroup::tile_load(matL_load, load_payload);

    // if constexpr (wg_size_x > 1)
    //   ctx.nbarrier.arrive();

    // now Sij is Pij
    subgroup::tile_broadcast_op<subgroup::tile_minus, matAcc_Sij_t>(
        *matAcc_Sij, matL_load.reg);

    matAcc_Sij->reg = xetla_exp<accum_t>(matAcc_Sij->reg);

    // if constexpr (wg_size_x > 1)
    //   ctx.nbarrier.wait();

    if constexpr (kIsDropout) {
      if (args.Dp_ptr) {
        using load_payload_mask_t = subgroup::mem_payload_t<
            mem_desc_t<
                uint8_t,
                mem_desc_Dp_Mask_t::layout,
                mem_desc_Dp_Mask_t::space>,
            dp_mask_tile_desc_t,
            subgroup::
                msg_type_v<dp_mask_tile_desc_t, mem_desc_Dp_Mask_t::space>,
            gpu_arch::Xe>;
        load_payload_mask_t load_payload_mask(ctx.mem_desc_Dpij);
        subgroup::tile_load(*mask_in, load_payload_mask);
        matAcc_Sij_drop->reg = matAcc_Sij->reg * mask_in->reg * args.dp_scale;
      } else {
        matAcc_Sij_drop->reg =
            ctx.dropout_op.template process<float>(matAcc_Sij->reg);
      }
    } else {
      matAcc_Sij_drop->reg = matAcc_Sij->reg;
    }

    // store Pij_drop to local memory, transpose it while saving
    using epilogue_p_t = group::epilogue_transp_t<
        group::epilogue_policy_tile_op<
            subgroup::chained_tile_op_t<>,
            gpu_arch::Xe>,
        tile_shape_BrBc,
        mem_desc_Pij_L_T_t>;
    epilogue_p_t epilogue;
    epilogue(ctx.g_brbc, *matAcc_Sij_drop, ctx.mem_desc_Pij_L_T);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
    ctx.nbarrier_y.arrive_wait();
  }

  // ======================= // gemm_dVj // ======================= //
  // Define kernel to compute dVj = Pij_dropped_T x dOi
  using gemm_dVj_t = group::gemm_t<
      compute_policy,
      tile_shape_BcHm,
      mem_desc_Pij_L_T_t,
      mem_desc_dOi_t>;
  using matAcc_dVj_t = typename gemm_dVj_t::matAcc_t;
  /// @brief gemm_dVj is used to compute dVj = P_dropped_ij_T x dOi
  /// # [Bc,Br] x [Br,H] = [Bc,H]
  inline void gemm_dVj(
      matAcc_dVj_t* matAcc_dVj,
      const arguments_t& args,
      uint32_t startF) {
    using gemm_args_t = typename gemm_dVj_t::arguments_t;

    uint32_t remainF = args.uF - startF;
    uint32_t boundary_k = remainF > kBr ? kBr : remainF;
    uint32_t loop_count = (boundary_k + accum_step - 1) / accum_step;

    // Gemm to compute dVj
    gemm_dVj_t gemm;
    gemm_args_t gemm_args(ctx.mem_desc_Pij_L_T, ctx.mem_desc_dOi, loop_count);
    gemm(
        ctx.g_bchm,
        *matAcc_dVj,
        gemm_args,
        0,
        /* nbarrier_base */ nbarrier_cnt);
  }

  // ======================= // gemm_dPij // ======================= //
  // Define kernel to compute dPij = dOi x Vj_T
  using gemm_dPij_t = group::
      gemm_t<compute_policy, tile_shape_BrBc, mem_desc_dOi_t, mem_desc_Vj_T_t>;
  using matAcc_dPij_t = typename gemm_dPij_t::matAcc_t;
  /// @brief gemm_dPij is used to compute dPij = dOi x Vj_T
  /// # [Br,H] x [H,Bc] = [Br,Bc]
  inline void gemm_dPij(
      matAcc_dPij_t* matAcc_dPij,
      dp_mask_tile_t* mask_in,
      const arguments_t& args) {
    using gemm_args_t = typename gemm_dPij_t::arguments_t;

    uint32_t loop_count = (args.uH + accum_step - 1) / accum_step;

    // Gemm to compute dPij
    gemm_dPij_t gemm;
    gemm_args_t gemm_args(ctx.mem_desc_dOi, ctx.mem_desc_Vj_T, loop_count);
    gemm(
        ctx.g_brbc,
        *matAcc_dPij,
        gemm_args,
        0,
        /* nbarrier_base */ nbarrier_cnt);

    // dropout bwd
    if constexpr (kIsDropout) {
      if (args.Dp_ptr) {
        matAcc_dPij->reg = matAcc_dPij->reg * mask_in->reg * args.dp_scale;
      } else {
        matAcc_dPij->reg =
            matAcc_dPij->reg * (1 - ctx.dropout_op.get_mask()) * args.dp_scale;
      }
    }
  }

  // ======================= // softmax_bwd // ======================= //
  /// @brief softmax_bwd is used to compute dSij = Pij * (dPij - Di)
  inline void softmax_bwd(
      matAcc_Sij_t* matAcc_dSij,
      matAcc_Sij_t* matAcc_Pij,
      matAcc_dPij_t* matAcc_dPij,
      const arguments_t& args) {
    using load_desc =
        subgroup::tile_desc_t<kSgBr, 1, kSgBr, 1, reg_layout::tiled>;
    using load_tile_t = subgroup::tile_t<accum_t, load_desc>;
    using load_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
        load_desc,
        msg_type::block_2d,
        gpu_arch::Xe>;

    load_tile_t matD_load;
    load_payload_t load_payload(ctx.mem_desc_Di);
    subgroup::tile_load(matD_load, load_payload);

    // if constexpr (wg_size_x > 1)
    //   ctx.nbarrier.arrive();

    subgroup::tile_broadcast_op<subgroup::tile_minus, matAcc_dPij_t>(
        *matAcc_dPij, matD_load.reg);

    matAcc_dSij->reg = matAcc_dPij->reg * matAcc_Pij->reg;

    // if constexpr (wg_size_x > 1)
    //   ctx.nbarrier.wait();

    // store dSij to local
    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<gpu_arch::Xe>,
        tile_shape_BrBc,
        mem_desc_dSij_L_t>;
    epilogue_t epilogue;
    epilogue(ctx.g_brbc, *matAcc_dSij, ctx.mem_desc_dSij_L);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
    ctx.nbarrier_y.arrive_wait();

    // Add bias if needed
    if constexpr (kUseBias) {
      // dSij = dBij
      using epilogue_t = group::epilogue_write_back_t<
          group::epilogue_policy_default<gpu_arch::Xe>,
          tile_shape_BrBc,
          mem_desc_dBij_t>;
      epilogue_t epilogue;
      epilogue(ctx.g_brbc, *matAcc_dSij, ctx.mem_desc_dBij);
    }
  }

  // ======================= // gemm_dQi // ======================= //
  // Define kernel to compute dQi = dQi + scale * (dSij x Kj)
  using gemm_dQi_t = group::
      gemm_t<compute_policy, tile_shape_BrHm, mem_desc_dSij_L_t, mem_desc_Kj_t>;
  using matAcc_dQi_t = typename gemm_dQi_t::matAcc_t;
  /// @brief gemm_dQi is used to compute dQi = dQi + scale * (dSij x Kj)
  /// # [Br,Bc] x [Bc,H] = [Br,H]
  inline void gemm_dQi(
      matAcc_dQi_t* matAcc_dQi,
      const arguments_t& args,
      uint32_t startT) {
    using gemm_args_t = typename gemm_dQi_t::arguments_t;

    uint32_t remainT = args.uT - startT;
    uint32_t boundary_k = remainT > kBc ? kBc : remainT;
    uint32_t loop_count = (boundary_k + accum_step - 1) / accum_step;

    // Gemm to compute dQi
    gemm_dQi_t gemm;
    gemm_args_t gemm_args(ctx.mem_desc_dSij_L, ctx.mem_desc_Kj, loop_count);
    gemm(
        ctx.g_brhm,
        *matAcc_dQi,
        gemm_args,
        0,
        /* nbarrier_base */ nbarrier_cnt);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
    ctx.nbarrier_y.arrive_wait();
  }

  // ======================= // gemm_dKj // ======================= //
  // Define kernel to compute dKj = dKj + dSij_T x Qi
  using gemm_dKj_t = group::gemm_t<
      compute_policy,
      tile_shape_BcHm,
      mem_desc_dSij_L_T_t,
      mem_desc_Qi_t>;
  using matAcc_dKj_t = typename gemm_dKj_t::matAcc_t;
  /// @brief gemm_dKj is used to compute dKj = dKj + dSij_T x Qi
  /// # [Bc,Br] x [Br,H] = [Bc,H]
  inline void gemm_dKj(
      matAcc_dKj_t* matAcc_dKj,
      matAcc_Sij_t* matAcc_dSij,
      const arguments_t& args,
      uint32_t startF) {
    // store dSij transpose to local
    using epilogue_s_t = group::epilogue_transp_t<
        group::epilogue_policy_tile_op<
            subgroup::chained_tile_op_t<>,
            gpu_arch::Xe>,
        tile_shape_BrBc,
        mem_desc_dSij_L_T_t>;
    epilogue_s_t epilogue;
    epilogue(ctx.g_brbc, *matAcc_dSij, ctx.mem_desc_dSij_L_T);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
    ctx.nbarrier_y.arrive_wait();

    using gemm_args_t = typename gemm_dKj_t::arguments_t;

    uint32_t remainF = args.uF - startF;
    uint32_t boundary_k = remainF > kBr ? kBr : remainF;
    uint32_t loop_count = (boundary_k + accum_step - 1) / accum_step;

    // Gemm to compute dKj
    gemm_dKj_t gemm;
    gemm_args_t gemm_args(ctx.mem_desc_dSij_L_T, ctx.mem_desc_Qi, loop_count);
    gemm(
        ctx.g_bchm,
        *matAcc_dKj,
        gemm_args,
        0,
        /* nbarrier_base */ nbarrier_cnt);
  }

  // ==================== // store dQ, dK and dV // ====================== //

  /// @brief store raw dQi to global memory.
  inline void store_dQi(matAcc_dQi_t& matAcc_dQi, const arguments_t& args) {
    using store_t = subgroup::mem_payload_t<
        mem_desc_dQi_acc_t,
        typename matAcc_dQi_t::tile_desc,
        msg_type::atomic_add,
        gpu_arch::Xe>;
    store_t dQ_store(ctx.mem_desc_dQi_acc);
    subgroup::tile_store(matAcc_dQi, dQ_store);
  }

  /// @brief store raw dKj to global memory.
  inline void store_dKj(matAcc_dKj_t& matAcc_dKj, const arguments_t& args) {
    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<gpu_arch::Xe>,
        tile_shape_BcHm,
        mem_desc_dKj_t>;
    epilogue_t epilogue;
    epilogue(ctx.g_bchm, matAcc_dKj, ctx.mem_desc_dKj);
  }

  /// @brief store raw dVj to global memory.
  inline void store_dVj(matAcc_dVj_t& matAcc_dVj, const arguments_t& args) {
    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<gpu_arch::Xe>,
        tile_shape_BcHm,
        mem_desc_dVj_t>;
    epilogue_t epilogue;
    epilogue(ctx.g_bchm, matAcc_dVj, ctx.mem_desc_dVj);
  }

  // ======================= // calc_D // ======================= //

  /// @brief calc_D is used to compute D = rowsum(dO * O)
  /// # [Br,H] * [Br,H] = [Br,H]
  inline void calc_D(
      matAcc_dQi_t* matAcc_O,
      matAcc_dQi_t* matAcc_dO,
      matAcc_dQi_t* matAcc_D,
      const arguments_t& args) {
    // define matD
    using store_desc =
        subgroup::tile_desc_t<kSgBr, 1, kSgBr, 1, reg_layout::tiled>;
    using store_tile_t = subgroup::tile_t<accum_t, store_desc>;
    using store_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
        store_desc,
        msg_type::block_2d,
        gpu_arch::Xe>;

    store_tile_t matD_store;

    // load matO and matdO
    using load_desc_O = gemm_dQi_t::matAcc_tile_desc_t;
    using load_tile_O_t = subgroup::tile_t<scalar_t, load_desc_O>;
    using load_payload_O_t = subgroup::mem_payload_t<
        mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>,
        load_desc_O,
        msg_type::block_2d,
        gpu_arch::Xe>;

    load_tile_O_t matO_load;
    load_payload_O_t load_payload_O(ctx.mem_desc_Oi);
    subgroup::tile_load(matO_load, load_payload_O);

    load_tile_O_t matdO_load;
    load_payload_O_t load_payload_dO(ctx.mem_desc_dOi);
    subgroup::tile_load(matdO_load, load_payload_dO);

    elemwise_cvt<matAcc_dQi_t, load_tile_O_t>(*matAcc_O, matO_load);
    elemwise_cvt<matAcc_dQi_t, load_tile_O_t>(*matAcc_dO, matdO_load);
    matAcc_D->reg = matAcc_O->reg * matAcc_dO->reg;

    using wg_row_sum_t =
        group_row_reduce_t<matAcc_dQi_t, wg_size_x, reduce_op::sum>;
    uint32_t reducer_slm =
        D_slm + ctx.sg_idy * wg_size_x * kSgBr * sizeof(accum_t);

    wg_row_sum_t wg_row_sum(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    matD_store.reg = wg_row_sum(*matAcc_D);

    // store matD
    store_payload_t store_payload_D(ctx.mem_desc_Di);
    if (ctx.sg_idx == 0) {
      subgroup::tile_store(matD_store, store_payload_D);
    }
  }

  // ======================= // convert_dQ // ======================= //

  /// @brief convert_dQ is used to convert dQ from fp32 to bf16
  inline void convert_dQ(const arguments_t& args) {
    // load matAcc_dQ
    using load_payload_t = subgroup::mem_payload_t<
        mem_desc_t<accum_t, mem_layout::row_major, mem_space::global>,
        typename gemm_dQi_t::matAcc_tile_desc_t,
        msg_type::block_2d,
        gpu_arch::Xe>;

    matAcc_dQi_t matdQ_load;
    // mat_dQi_t matdQ;

    load_payload_t load_payload(ctx.mem_desc_dQi_acc);
    subgroup::tile_load(matdQ_load, load_payload);

    matdQ_load.reg *= args.sm_scale;

    // subgroup::elemwise_cvt(matdQ, matdQ_load);

    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<gpu_arch::Xe>,
        tile_shape_BrHm,
        mem_desc_dQi_t>;
    epilogue_t epilogue;
    epilogue(ctx.g_brhm, matdQ_load, ctx.mem_desc_dQi);
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile
  /// time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count_Sij = gemm_Sij_t::barrier_count;
    constexpr uint32_t barrier_count_dVj = gemm_dVj_t::barrier_count;
    constexpr uint32_t count =
        std::max(barrier_count_Sij, barrier_count_dVj) + nbarrier_cnt;
    static_assert(
        count <= 32, "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_Pij + slm_size_D + slm_size_Sij;
    static_assert(
        size <= (128 * 1024),
        "The local memory size should be less than 128KB!");
    return size;
  }

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(uint32_t total_batches) {
    // local range
    sycl::range<3> local_range = sycl::range<3>{1, wg_size_y, wg_size_x};
    // group range
    sycl::range<3> group_range = sycl::range<3>{total_batches, 1, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  }

  // ================= // Entry of the functor // ================= //

  inline KERNEL_FUNC void operator()(
      const nd_item<3>& item,
      const arguments_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // init context
    ctx.init_context(item, args);

    // Calc D
    matAcc_dQi_t matAcc_O, matAcc_dO, matAcc_D;
    for (uint32_t startF = 0; startF < args.uF; startF += kBr) {
      ctx.update_context_D(item, args, startF);
      calc_D(&matAcc_O, &matAcc_dO, &matAcc_D, args);
    }

    matAcc_dKj_t matAcc_dKj;
    matAcc_dVj_t matAcc_dVj;
    for (uint32_t startT = 0; startT < args.uT; startT += kBc) {
      // update context for KV
      ctx.update_context_KV(item, args, startT);
      // init dK dV
      matAcc_dKj.init(0);
      matAcc_dVj.init(0);
      for (uint32_t startF = 0; startF < args.uF; startF += kBr) {
        if constexpr (kIsCausal) {
          if (startT >= std::min(startF + kBr, args.uF))
            continue;
        }
        // update context for Q
        ctx.update_context_Q(item, args, startF, startT);
        // compute fwd Sij
        matAcc_Sij_t matAcc_Sij(0);
        gemm_Sij(&matAcc_Sij, args);
        // apply mask
        apply_mask(&matAcc_Sij, args, startF, startT);
        // softmax_fwd
        matAcc_Sij_t matAcc_Sij_drop(0);
        dp_mask_tile_t mask_in;
        softmax_fwd(&matAcc_Sij, &matAcc_Sij_drop, &mask_in, args);

        // compute dVj
        gemm_dVj(&matAcc_dVj, args, startF);
        // compute dPij
        matAcc_dPij_t matAcc_dPij(0);
        gemm_dPij(&matAcc_dPij, &mask_in, args);

        // softmax_bwd
        matAcc_Sij_t matAcc_dSij(0);
        softmax_bwd(&matAcc_dSij, &matAcc_Sij, &matAcc_dPij, args);

        // compute dQi
        matAcc_dQi_t matAcc_dQi(0);
        gemm_dQi(&matAcc_dQi, args, startT);
        store_dQi(matAcc_dQi, args);
        // compute dKj
        gemm_dKj(&matAcc_dKj, &matAcc_dSij, args, startF);
      }
      // store Kj and Vj
      matAcc_dKj.reg = args.sm_scale * matAcc_dKj.reg;
      store_dKj(matAcc_dKj, args);
      store_dVj(matAcc_dVj, args);
    }

    for (uint32_t startF = 0; startF < args.uF; startF += kBr) {
      ctx.update_context_dQ(item, args, startF);
      convert_dQ(args);
    }
  }
}; // fmha_backward_t

template <typename fmha_backward_op_t, typename T>
struct FmhaBackwardKernelFunctor {
  KERNEL_MAIN void operator()(sycl::nd_item<3> item) const {
    // init fmha forward op and arguments
    fmha_backward_op_t fmha_bwd_op;
    using accscalar_t = fmha_backward_op_t::accum_t;
    typename fmha_backward_op_t::arguments_t args(
        grad_out,
        query,
        key,
        value,
        bias,
        dropout,
        out,
        (accscalar_t*)log_sumexp,
        (accscalar_t*)workspace,
        (accscalar_t*)grad_q_tmp,
        (accscalar_t)dropout_prob,
        grad_query,
        grad_key,
        grad_value,
        grad_bias,
        num_batches,
        num_heads,
        head_size,
        num_queries,
        num_keys,
        bias_strideB,
        bias_strideN,
        bias_strideF,
        attn_mask_padding,
        (accscalar_t)alpha,
        seed,
        offset);

    // call the functor
    fmha_bwd_op(item, args);
  }
  FmhaBackwardKernelFunctor(
      T* grad_out_,
      T* query_,
      T* key_,
      T* value_,
      T* bias_,
      uint8_t* dropout_,
      T* out_,
      void* log_sumexp_,
      void* workspace_,
      void* grad_q_tmp_,
      float dropout_prob_,
      T* grad_query_,
      T* grad_key_,
      T* grad_value_,
      T* grad_bias_,
      uint32_t num_batches_,
      uint32_t num_heads_,
      uint32_t head_size_,
      uint32_t num_queries_,
      uint32_t num_keys_,
      uint32_t bias_strideB_,
      uint32_t bias_strideN_,
      uint32_t bias_strideF_,
      uint32_t attn_mask_padding_,
      float alpha_,
      uint64_t seed_,
      uint64_t offset_)
      : grad_out(grad_out_),
        query(query_),
        key(key_),
        value(value_),
        bias(bias_),
        dropout(dropout_),
        out(out_),
        log_sumexp(log_sumexp_),
        workspace(workspace_),
        grad_q_tmp(grad_q_tmp_),
        dropout_prob(dropout_prob_),
        grad_query(grad_query_),
        grad_key(grad_key_),
        grad_value(grad_value_),
        grad_bias(grad_bias_),
        num_batches(num_batches_),
        num_heads(num_heads_),
        head_size(head_size_),
        num_queries(num_queries_),
        num_keys(num_keys_),
        bias_strideB(bias_strideB_),
        bias_strideN(bias_strideN_),
        bias_strideF(bias_strideF_),
        attn_mask_padding(attn_mask_padding_),
        alpha(alpha_),
        seed(seed_),
        offset(offset_) {}

 private:
  T* grad_out;
  T* query;
  T* key;
  T* value;
  T* bias;
  uint8_t* dropout;
  T* out;
  void* log_sumexp;
  void* workspace;
  void* grad_q_tmp;
  float dropout_prob;
  T* grad_query;
  T* grad_key;
  T* grad_value;
  T* grad_bias;
  uint32_t num_batches;
  uint32_t num_heads;
  uint32_t head_size;
  uint32_t num_queries;
  uint32_t num_keys;
  uint32_t bias_strideB;
  uint32_t bias_strideN;
  uint32_t bias_strideF;
  uint32_t attn_mask_padding;
  float alpha;
  uint64_t seed;
  uint64_t offset;
};

// The launcher of fmha backward kernel
template <
    typename fmha_policy,
    typename T,
    bool kUseBias,
    bool kIsCausal,
    bool kIsDropout>
void xetla_fmha_backward_kernel(
    sycl::queue& q,
    T* grad_out,
    T* query,
    T* key,
    T* value,
    T* bias,
    uint8_t* dropout,
    T* out,
    void* log_sumexp,
    void* workspace,
    void* grad_q_tmp,
    float alpha,
    float dropout_prob,
    T* grad_query,
    T* grad_key,
    T* grad_value,
    T* grad_bias,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t bias_strideB,
    uint32_t bias_strideN,
    uint32_t bias_strideF,
    uint32_t attn_mask_padding,
    uint64_t seed,
    uint64_t offset) {
#ifdef SDP_DBG
  printf(
      "B, N, F, T, H, MT, strideB, strideN, strideF: %d, %d, %d, %d, %d, %d, %d, %d, %d, UseBias: %d, IsCausal: %d, IsDropout: %d, sm_scale: %f\n",
      num_batches,
      num_heads,
      num_queries,
      num_keys,
      head_size,
      attn_mask_padding,
      bias_strideB,
      bias_strideN,
      bias_strideF,
      kUseBias,
      kIsCausal,
      kIsDropout,
      alpha);
#endif
  // fmha backward kernel
  using fmha_backward_op_t =
      fmha_backward_t<fmha_policy, T, kUseBias, kIsCausal, kIsDropout>;

  sycl::nd_range<3> NdRange =
      fmha_backward_op_t::get_nd_range(num_batches * num_heads);

  auto cgf = DPCPP_Q_CGF(cgh) {
    FmhaBackwardKernelFunctor<fmha_backward_op_t, T> kfn(
        grad_out,
        query,
        key,
        value,
        bias,
        dropout,
        out,
        log_sumexp,
        workspace,
        grad_q_tmp,
        dropout_prob,
        grad_query,
        grad_key,
        grad_value,
        grad_bias,
        num_batches,
        num_heads,
        head_size,
        num_queries,
        num_keys,
        bias_strideB,
        bias_strideN,
        bias_strideF,
        attn_mask_padding,
        alpha,
        seed,
        offset);
    cgh.parallel_for<decltype(kfn)>(NdRange, kfn);
  };
  DPCPP_Q_SUBMIT(q, cgf);
}
} // namespace fmha
#define CALL_IMPL_FUNC(P)                                                  \
  fmha::xetla_fmha_backward_kernel<P, T, kUseBias, kIsCausal, kIsDropout>( \
      q,                                                                   \
      grad_out,                                                            \
      query,                                                               \
      key,                                                                 \
      value,                                                               \
      bias,                                                                \
      dropout,                                                             \
      out,                                                                 \
      log_sumexp,                                                          \
      workspace,                                                           \
      grad_q_tmp,                                                          \
      alpha,                                                               \
      dropout_prob,                                                        \
      grad_query,                                                          \
      grad_key,                                                            \
      grad_value,                                                          \
      grad_bias,                                                           \
      num_batches,                                                         \
      num_heads,                                                           \
      head_size,                                                           \
      num_queries,                                                         \
      num_keys,                                                            \
      bias_strideB,                                                        \
      bias_strideN,                                                        \
      bias_strideF,                                                        \
      attn_mask_padding,                                                   \
      seed,                                                                \
      offset)

/// @brief Main execution function for flash mha forward.
template <
    typename T,
    bool kUseBias = false,
    bool kIsCausal = false,
    bool kIsDropout = false>
void fmha_backward_kernel_policy(
    sycl::queue& q,
    T* grad_out,
    T* query,
    T* key,
    T* value,
    T* bias,
    uint8_t* dropout,
    T* out,
    void* log_sumexp,
    void* workspace,
    void* grad_q_tmp,
    float alpha,
    float dropout_prob,
    T* grad_query,
    T* grad_key,
    T* grad_value,
    T* grad_bias,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t bias_strideB,
    uint32_t bias_strideN,
    uint32_t bias_strideF,
    uint32_t attn_mask_padding,
    uint64_t seed = 0,
    uint64_t offset = 123) {
  if (head_size <= 64) {
    CALL_IMPL_FUNC(fmha_bwd_policy_128x128x64);
  } else if (head_size <= 128) {
    CALL_IMPL_FUNC(fmha_bwd_policy_128x128x128);
  } else if (head_size <= 256) {
    CALL_IMPL_FUNC(fmha_bwd_policy_128x128x256);
  } else if (head_size <= 512) {
    CALL_IMPL_FUNC(fmha_bwd_policy_64x128x512);
  } else {
    assert(false);
  }
}

#undef CALL_IMPL_FUNC

template <typename T, bool... Bs>
void dispatch_fmha_backward(
    sycl::queue& q,
    T* grad_out,
    T* query,
    T* key,
    T* value,
    T* bias,
    uint8_t* dropout,
    T* out,
    void* log_sumexp,
    void* workspace,
    void* grad_q_tmp,
    float alpha,
    float dropout_prob,
    T* grad_query,
    T* grad_key,
    T* grad_value,
    T* grad_bias,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t bias_strideB,
    uint32_t bias_strideN,
    uint32_t bias_strideF,
    uint32_t attn_mask_padding,
    uint64_t seed,
    uint64_t offset) {
  fmha_backward_kernel_policy<T, Bs...>(
      q,
      grad_out,
      query,
      key,
      value,
      bias,
      dropout,
      out,
      log_sumexp,
      workspace,
      grad_q_tmp,
      alpha,
      dropout_prob,
      grad_query,
      grad_key,
      grad_value,
      grad_bias,
      num_batches,
      num_heads,
      head_size,
      num_queries,
      num_keys,
      bias_strideB,
      bias_strideN,
      bias_strideF,
      attn_mask_padding,
      seed,
      offset);
}

// dispatch different conditions
template <typename T, bool... Bs, typename... Ts>
void dispatch_fmha_backward(
    sycl::queue& q,
    T* grad_out,
    T* query,
    T* key,
    T* value,
    T* bias,
    uint8_t* dropout,
    T* out,
    void* log_sumexp,
    void* workspace,
    void* grad_q_tmp,
    float alpha,
    float dropout_prob,
    T* grad_query,
    T* grad_key,
    T* grad_value,
    T* grad_bias,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t bias_strideB,
    uint32_t bias_strideN,
    uint32_t bias_strideF,
    uint32_t attn_mask_padding,
    uint64_t seed,
    uint64_t offset,
    bool b,
    Ts... ts) {
  if (b) {
    dispatch_fmha_backward<T, Bs..., true>(
        q,
        grad_out,
        query,
        key,
        value,
        bias,
        dropout,
        out,
        log_sumexp,
        workspace,
        grad_q_tmp,
        alpha,
        dropout_prob,
        grad_query,
        grad_key,
        grad_value,
        grad_bias,
        num_batches,
        num_heads,
        head_size,
        num_queries,
        num_keys,
        bias_strideB,
        bias_strideN,
        bias_strideF,
        attn_mask_padding,
        seed,
        offset,
        ts...);
  } else {
    dispatch_fmha_backward<T, Bs..., false>(
        q,
        grad_out,
        query,
        key,
        value,
        bias,
        dropout,
        out,
        log_sumexp,
        workspace,
        grad_q_tmp,
        alpha,
        dropout_prob,
        grad_query,
        grad_key,
        grad_value,
        grad_bias,
        num_batches,
        num_heads,
        head_size,
        num_queries,
        num_keys,
        bias_strideB,
        bias_strideN,
        bias_strideF,
        attn_mask_padding,
        seed,
        offset,
        ts...);
  }
}

template <typename T>
void fmha_backward_kernel_impl(
    sycl::queue& q,
    T* grad_out,
    T* query,
    T* key,
    T* value,
    T* bias,
    uint8_t* dropout,
    T* out,
    void* log_sumexp,
    void* workspace,
    void* grad_q_tmp,
    float alpha,
    float dropout_prob,
    T* grad_query,
    T* grad_key,
    T* grad_value,
    T* grad_bias,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t bias_strideB,
    uint32_t bias_strideN,
    uint32_t bias_strideF,
    uint32_t attn_mask_padding,
    bool is_causal,
    bool is_dropout,
    uint64_t seed_t,
    uint64_t offset_t) {
  bool is_bias = bias ? true : false;

  dispatch_fmha_backward<T>(
      q,
      grad_out,
      query,
      key,
      value,
      bias,
      dropout,
      out,
      log_sumexp,
      workspace,
      grad_q_tmp,
      alpha,
      dropout_prob,
      grad_query,
      grad_key,
      grad_value,
      grad_bias,
      num_batches,
      num_heads,
      head_size,
      num_queries,
      num_keys,
      bias_strideB,
      bias_strideN,
      bias_strideF,
      attn_mask_padding,
      seed_t,
      offset_t,
      is_bias,
      is_causal,
      is_dropout);
}

// dispatch datatype
void fmha_backward_kernel(
    XetlaType xeType,
    sycl::queue& q,
    void* grad_out,
    void* query,
    void* key,
    void* value,
    void* bias,
    void* dropout,
    void* out,
    void* log_sumexp,
    void* workspace,
    void* grad_q_tmp,
    float alpha,
    float dropout_prob,
    void* grad_query,
    void* grad_key,
    void* grad_value,
    void* grad_bias,
    uint32_t num_batches,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t num_queries,
    uint32_t num_keys,
    uint32_t bias_strideB,
    uint32_t bias_strideN,
    uint32_t bias_strideF,
    uint32_t attn_mask_padding,
    bool is_causal,
    bool is_dropout,
    uint64_t seed_t = 0,
    uint64_t offset_t = 123) {
  if (xeType == XetlaType::fp16) {
    fmha_backward_kernel_impl<fp16>(
        q,
        (fp16*)grad_out,
        (fp16*)query,
        (fp16*)key,
        (fp16*)value,
        (fp16*)bias,
        (uint8_t*)dropout,
        (fp16*)out,
        log_sumexp,
        workspace,
        grad_q_tmp,
        alpha,
        dropout_prob,
        (fp16*)grad_query,
        (fp16*)grad_key,
        (fp16*)grad_value,
        (fp16*)grad_bias,
        num_batches,
        num_heads,
        head_size,
        num_queries,
        num_keys,
        bias_strideB,
        bias_strideN,
        bias_strideF,
        attn_mask_padding,
        is_causal,
        is_dropout,
        seed_t,
        offset_t);
  } else {
    fmha_backward_kernel_impl<bf16>(
        q,
        (bf16*)grad_out,
        (bf16*)query,
        (bf16*)key,
        (bf16*)value,
        (bf16*)bias,
        (uint8_t*)dropout,
        (bf16*)out,
        log_sumexp,
        workspace,
        grad_q_tmp,
        alpha,
        dropout_prob,
        (bf16*)grad_query,
        (bf16*)grad_key,
        (bf16*)grad_value,
        (bf16*)grad_bias,
        num_batches,
        num_heads,
        head_size,
        num_queries,
        num_keys,
        bias_strideB,
        bias_strideN,
        bias_strideF,
        attn_mask_padding,
        is_causal,
        is_dropout,
        seed_t,
        offset_t);
  }
}
} // namespace gpu::xetla
