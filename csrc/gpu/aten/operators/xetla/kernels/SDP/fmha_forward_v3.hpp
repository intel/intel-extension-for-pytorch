
#pragma once
/*
Fused Multi-Head Attention Forward

This is an implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf)
*/
#include <sys/types.h>
#include <limits>
#include "fmha_forward_policy.h"
#include "fmha_utils.h"

namespace gpu::xetla {
namespace fmha {

template <
    typename fmha_policy,
    typename scalar_t,
    gpu_arch arch_tag,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsCausal,
    bool kSeqLast,
    bool kIsTraining,
    bool kIsDropout,
    bool kVarlen,
    bool kIsLocal,
    uint32_t kHeadPerKv>
class fmha_forward_v3_t {
 public:
  using accum_t = float;
  static constexpr accum_t kNegInfinity =
      -std::numeric_limits<accum_t>::infinity();

  struct arguments_t {
    // Input tensors
    scalar_t* Q_ptr; // [num_tokens, num_heads_query, head_size]
    scalar_t* K_ptr; // [num_blocks, block_size, num_heads_kv, head_size]
    scalar_t* V_ptr; // [num_blocks, block_size, num_heads_kv, head_size]
    scalar_t* A_ptr =
        nullptr; // [B, N, 1, T] - Alibi | [B, N] or [N] - alibi_slopes(float)
    scalar_t* S_ptr; // [batch_size, num_heads_query] - attention bias
    scalar_t* B_ptr = nullptr; // [1/B, 1/N, 1/F, M] - bias
    uint8_t* Dp_ptr = nullptr; // [B, N, F, T] - dropout mask
    // Output tensor
    scalar_t* O_ptr; // raw: [B, F, N, H]; permute: [B, N, F, H] - output
    accum_t* L_ptr; // logsum softmax: [B, N, F]
    // Dimension size
    uint32_t uB;
    uint32_t uN;
    uint32_t uNkv;
    uint32_t uH;
    uint32_t uF;
    uint32_t uT;
    uint32_t q_strideB;
    uint32_t q_strideN;
    uint32_t q_strideF;
    uint32_t kv_strideB;
    uint32_t kv_strideN;
    uint32_t kv_strideT;
    uint32_t bias_strideB;
    uint32_t bias_strideN;
    uint32_t bias_strideF;
    // Sequence length info
    int32_t* cu_seqlen_q;
    int32_t* cu_seqlen_k;
    // Softmax scale is the reciprocal square root of head size by default
    accum_t sm_scale;
    // Dropout scale is computed from dropout prob
    accum_t dp_prob;
    accum_t dp_scale;
    uint32_t uAT;
    uint32_t uMT;

    // sliding window size
    int32_t w_left, w_right;

    uint64_t seed;
    uint64_t offset;
    bool is_bias_add;
    accum_t softcap;
    int32_t* block_tables; // [B, max_blocks_per_seq]
    uint32_t max_blocks_per_seq;
    uint32_t block_size;

    inline arguments_t() = default;
    inline arguments_t(
        scalar_t* query,
        scalar_t* key,
        scalar_t* value,
        scalar_t* alibi,
        scalar_t* sink,
        scalar_t* bias,
        uint8_t* dropout,
        scalar_t* out,
        accum_t* logsumsoftmax,
        uint32_t num_batches,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_size,
        uint32_t num_queries,
        uint32_t num_keys,
        uint32_t q_strideB,
        uint32_t q_strideN,
        uint32_t q_strideF,
        uint32_t kv_strideB,
        uint32_t kv_strideN,
        uint32_t kv_strideT,
        uint32_t bias_strideB,
        uint32_t bias_strideN,
        uint32_t bias_strideF,
        int32_t* cu_seqlen_q,
        int32_t* cu_seqlen_k,
        accum_t sm_scale,
        accum_t dropout_prob,
        uint32_t alibi_padded_block_size,
        uint32_t attn_mask_padded_block_size,
        int32_t window_size_left,
        int32_t window_size_right,
        uint64_t seed_t,
        uint64_t offset_t,
        accum_t softcap = -1.,
        int32_t* block_tables = nullptr,
        uint32_t max_blocks_per_seq = 0,
        uint32_t block_size = 0)
        : Q_ptr(query),
          K_ptr(key),
          V_ptr(value),
          A_ptr(alibi),
          S_ptr(sink),
          B_ptr(bias),
          Dp_ptr(dropout),
          O_ptr(out),
          L_ptr(logsumsoftmax),
          uB(num_batches),
          uN(num_heads),
          uNkv(num_kv_heads),
          uH(head_size),
          uF(num_queries),
          uT(num_keys),
          q_strideB(q_strideB),
          q_strideN(q_strideN),
          q_strideF(q_strideF),
          kv_strideB(kv_strideB),
          kv_strideN(kv_strideN),
          kv_strideT(kv_strideT),
          bias_strideB(bias_strideB),
          bias_strideN(bias_strideN),
          bias_strideF(bias_strideF),
          cu_seqlen_q(cu_seqlen_q),
          cu_seqlen_k(cu_seqlen_k),
          sm_scale(sm_scale),
          dp_prob(dropout_prob),
          dp_scale(1.f / (1.f - dropout_prob)),
          uAT(alibi_padded_block_size),
          uMT(attn_mask_padded_block_size),
          w_left(window_size_left),
          w_right(window_size_right),
          seed(seed_t),
          offset(offset_t),
          is_bias_add(bias_strideF == 0),
          softcap(softcap),
          block_tables(block_tables),
          max_blocks_per_seq(max_blocks_per_seq),
          block_size(block_size) {}
  };

 private:
  // -------------------- // Compute policy // -------------------- //
  static constexpr uint32_t accum_step = fmha_policy::accum_step;
  static constexpr uint32_t stages =
      fmha_perf_knob_t<arch_tag>::prefetch_distance;
  static constexpr uint32_t sync_freq =
      fmha_perf_knob_t<arch_tag>::periodic_sync_interval;

  using comp_attr = group::compute_attr_t<scalar_t, scalar_t, accum_t>;
  using knobs = group::perf_tuning_knob_t<accum_step, stages, sync_freq>;
  using compute_policy_BrBc =
      group::compute_policy_default_multi_xmx<comp_attr, knobs, arch_tag>;
  // TODO: add k slicing
  using compute_policy_BrBm =
      group::compute_policy_default_multi_xmx<comp_attr, knobs, arch_tag>;
  using tanh_t = typename subgroup::tanh_op_t;
  // ---------------- // Tile shape and Threads // ---------------- //
  static constexpr uint32_t kBr = fmha_policy::kBr;
  static constexpr uint32_t kBc = fmha_policy::kBc;
  static constexpr uint32_t kHm = fmha_policy::kHm;
  static constexpr uint32_t kSgBr = fmha_policy::kSgBr;
  static constexpr uint32_t kSgBc = fmha_policy::kSgBc;
  static constexpr uint32_t kSgHm = fmha_policy::kSgHm;

  using tile_shape_BrBc = group::tile_shape_t<kBc, kBr, kSgBc, kSgBr>;
  using tile_shape_BrHm = group::tile_shape_t<kHm, kBr, kSgHm, kSgBr>;

  static constexpr uint32_t wg_size_x = tile_shape_BrBc::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_BrBc::wg_size_y;
  using work_group_t = typename tile_shape_BrBc::work_group_t;
  static constexpr uint32_t wg_size = work_group_t::size;

  static_assert(
      kHm / kSgHm == kBc / kSgBc,
      "wg_size_x must be the same between Hm and Bc");
  static_assert(
      wg_size <= arch_attr_t<arch_tag>::thread_per_wg,
      "The number of threads should be less than threads in one workgroup!");
  static_assert(
      arch_has_xmx<arch_tag>,
      "fmha_forward_v3_t requires XMX architecture for now!");

  // --------------------- // Memory desc // ---------------------- //
  // suffix: L -> local; T -> transpose
  using mem_desc_Qi_t =
      mem_mask_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Qi_L_t =
      mem_mask_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_Kj_T_t =
      mem_mask_desc_t<scalar_t, mem_layout::col_major, mem_space::global>;
  using mem_desc_Pij_L_t =
      mem_mask_desc_t<scalar_t, mem_layout::row_major, mem_space::local>;
  using mem_desc_Vj_t =
      mem_mask_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Bij_t =
      mem_mask_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Oi_t =
      mem_mask_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;
  using mem_desc_Ai_t =
      mem_mask_desc_t<scalar_t, mem_layout::row_major, mem_space::global>;

  using mem_desc_Li_t =
      mem_mask_desc_t<accum_t, mem_layout::row_major, mem_space::global>;

  using mem_desc_Dp_Mask_t =
      mem_mask_desc_t<uint8_t, mem_layout::row_major, mem_space::global>;

  // ------------------- // Slm and nbarrier // ------------------- //
  static constexpr uint32_t slm_size_Qi =
      kHeadPerKv * kBr * kHm * sizeof(scalar_t);
  static constexpr uint32_t slm_size_Pij =
      kHeadPerKv * kBr * kBc * sizeof(scalar_t);
  static constexpr uint32_t slm_size_softmax =
      kHeadPerKv * ((wg_size_x > 1) ? wg_size * kSgBr * sizeof(accum_t) : 0);
  // Slm addr to store inermediate results
  static constexpr uint32_t Qi_slm = 0;
  static constexpr uint32_t Pij_slm = Qi_slm + slm_size_Qi;
  static constexpr uint32_t softmax_slm = Pij_slm + slm_size_Pij;

  using gemm_Sij_t = group::gqa_t<
      compute_policy_BrBc,
      tile_shape_BrBc,
      mem_desc_Qi_L_t,
      mem_desc_Kj_T_t,
      kHeadPerKv>;
  using matAccSij_t = typename gemm_Sij_t::matAcc_t;
  using dropout_fd_t = dropout_fwd_t<matAccSij_t::tile_elems>;
  using dp_mask_tile_desc_t = typename gemm_Sij_t::matAcc_tile_desc_t;
  using dp_mask_tile_t = subgroup::tile_t<uint8_t, dp_mask_tile_desc_t>;

  // ======================== // Context // ======================= //

  /// @brief Used to store variables in the flash mha loops
  struct context_t {
    // thread id
    work_group_t g;
    uint32_t sg_idx;
    uint32_t sg_idy;
    // nbarrier
    xetla_nbarrier_t<wg_size_x, wg_size_x, gpu_arch::XeLpg> nbarrier;
    // softmax statistics
    std::array<xetla_vector<accum_t, kSgBr>, kHeadPerKv> softmax_m;
    std::array<xetla_vector<accum_t, kSgBr>, kHeadPerKv> softmax_l;
    // mem desc variables
    std::array<mem_desc_Qi_t, kHeadPerKv> mem_desc_Qi;
    std::array<mem_desc_Qi_L_t, kHeadPerKv> mem_desc_Qi_L;
    mem_desc_Kj_T_t mem_desc_Kj_T;
    std::array<mem_desc_Pij_L_t, kHeadPerKv> mem_desc_Pij_L;
    mem_desc_Vj_t mem_desc_Vj;
    mem_desc_Bij_t mem_desc_Bij;
    std::array<mem_desc_Oi_t, kHeadPerKv> mem_desc_Oi;
    mem_desc_Ai_t mem_desc_Ai;
    mem_desc_Li_t mem_desc_Li;
    mem_desc_Dp_Mask_t mem_desc_Dpij;
    dropout_fd_t dropout_op;

    uint32_t gid;
    uint32_t batch_id;
    uint32_t head_per_kv;
    uint32_t head_id_start;
    uint32_t head_id_kv;

    int32_t kv_offset_y;
    int32_t kv_offset_x;

    // local attention
    uint32_t local_left;
    uint32_t local_right;

    inline context_t() = default;

    /// @brief Initialize invariant variables in the flash mha loop
    inline void init_context(nd_item<3>& item, arguments_t& args) {
      // thread id
      uint32_t sg_id = item.get_local_linear_id();
      g.init(sg_id);
      sg_idx = sg_id % wg_size_x;
      sg_idy = sg_id / wg_size_x;
      // nbarrier
      nbarrier.init_nbarrier(sg_idy, nbarrier_role::producer_consumer);

      // mem desc variables
      gid = item.get_group(0);
      batch_id = gid / args.uNkv; // get batch idx
      head_id_kv = gid % args.uNkv; // get kv head idx
      head_per_kv = args.uN / args.uNkv; // number of heads per kv head
      assert(head_per_kv == kHeadPerKv);
      head_id_start = head_id_kv; // get head idx for kv

      kv_offset_y =
          (batch_id * args.kv_strideB + head_id_kv * args.kv_strideN) /
          args.kv_strideT;
      kv_offset_x =
          args.kv_strideT > args.kv_strideN ? head_id_kv * args.kv_strideN : 0;

      softmax_l.fill(0.f);
      softmax_m.fill(kNegInfinity);

      if constexpr (kSeqLast) { // 2d mem: [F, BxNxH]
        // startF
        int32_t start_y = item.get_group(1) * kBr;
        uint32_t end_y = start_y + kBr;
        // boundaryF
        uint32_t boundary_y = args.uF;
        end_y = end_y > boundary_y ? boundary_y : end_y;

        // XXX: code-gen crash
        uint32_t b_stride = args.uN * args.uH;

        for (uint32_t i = 0; i < kHeadPerKv; i++) {
          auto head_id = head_id_start * kHeadPerKv + i;
          int32_t start_acc = batch_id * b_stride + head_id * args.uH;
          uint32_t end_x = start_acc + args.uH;
          mem_desc_Qi[i].init(
              args.Q_ptr, {end_x, end_y, args.q_strideF}, {start_acc, start_y});
          mem_desc_Oi[i].init(
              args.O_ptr,
              {end_x, end_y, b_stride * args.uB},
              {start_acc, start_y});
        }
      } else if constexpr (kVarlen) {
        int32_t start_y = args.cu_seqlen_q[batch_id] + item.get_group(1) * kBr;
        uint32_t end_y = start_y + kBr;
        int32_t limit_y = args.cu_seqlen_q[batch_id + 1];
        end_y = end_y < limit_y ? end_y : limit_y;

        const uint32_t ld_qo = args.uH * args.uN;

        for (uint32_t i = 0; i < kHeadPerKv; i++) {
          auto head_id = head_id_start * kHeadPerKv + i;
          int32_t start_acc = head_id * args.uH;
          uint32_t end_x = start_acc + args.uH;
          mem_desc_Qi[i].init(
              args.Q_ptr, {end_x, end_y, ld_qo}, {start_acc, start_y});
          mem_desc_Oi[i].init(
              args.O_ptr, {end_x, end_y, ld_qo}, {start_acc, start_y});
        }
        // get current kv location
        kv_offset_y = 0;
        for (int32_t i = 0; i <= static_cast<int>(batch_id) - 1; ++i) {
          kv_offset_y += args.cu_seqlen_k[i];
        }

        // for local attention
        if constexpr (kIsLocal) {
          if constexpr (kIsCausal) {
            args.w_right = 0;
          }
          int32_t startF = item.get_group(1) * kBr;
          uint32_t real_T = args.cu_seqlen_k[batch_id];
          uint32_t real_F =
              args.cu_seqlen_q[batch_id + 1] - args.cu_seqlen_q[batch_id];
          uint32_t seq_diff = real_T - real_F;
          local_left = args.w_left == -1
              ? 0
              : std::max(0, int(seq_diff + startF - args.w_left));
          local_right = args.w_right == -1
              ? real_T - 1
              : std::min(
                    seq_diff + startF + kBr + args.w_right - 1, real_T - 1);
        }
      } else { // 2d mem: [BxF, NxH]
        for (uint32_t i = 0; i < kHeadPerKv; i++) {
          auto head_id = head_id_start * kHeadPerKv + i;
          int32_t ptr_offset_y =
              (batch_id * args.q_strideB + head_id * args.q_strideN) /
              args.q_strideF;
          int32_t ptr_offset_x =
              args.q_strideF > args.q_strideN ? head_id * args.q_strideN : 0;

          // startF
          int32_t start_y = ptr_offset_y + item.get_group(1) * kBr;
          uint32_t end_y = start_y + kBr;
          // boundaryF
          uint32_t boundary_y = ptr_offset_y + args.uF;
          end_y = end_y > boundary_y ? boundary_y : end_y;

          int32_t start_acc = ptr_offset_x;
          uint32_t end_acc = start_acc + args.uH;

          mem_desc_Qi[i].init(
              args.Q_ptr,
              {end_acc, end_y, args.q_strideF},
              {start_acc, start_y});
          mem_desc_Oi[i].init(
              args.O_ptr,
              {end_acc, end_y, args.q_strideF},
              {start_acc, start_y});
        }
      }

      int32_t start_x_ml = item.get_group(1) * kBr + sg_idy * kSgBr;
      int32_t start_y_ml = gid;
      mem_desc_Li.init(
          args.L_ptr,
          {args.uF, args.uB * args.uN, args.uF},
          {start_x_ml, start_y_ml});
      for (uint32_t i = 0; i < kHeadPerKv; i++) {
        mem_desc_Qi_L[i].init(
            Qi_slm + i * kBr * kHm * sizeof(scalar_t), {kHm, kBr, kHm}, {0, 0});
        mem_desc_Pij_L[i].init(
            Pij_slm + i * kBr * kBc * sizeof(scalar_t),
            {kBc, kBr, kBc},
            {0, 0});
      }
    }

    /// @brief Update variables for each flash mha loop
    inline void update_context(arguments_t& args, uint32_t startT) {
      // TODO: what's this startT for

      if constexpr (kSeqLast) {
        int32_t start_x = startT;
        uint32_t end_x = start_x + kBc;
        uint32_t boundary_x = args.uT;
        end_x = end_x > boundary_x ? boundary_x : end_x;

        // XXX: code-gen crash
        uint32_t b_stride = args.uNkv * args.uH;
        int32_t start_acc = batch_id * b_stride + head_id_kv * args.uH;
        uint32_t end_y = start_acc + args.uH;

        mem_desc_Kj_T.init(
            args.K_ptr,
            {end_x, end_y, b_stride * args.uB},
            {start_x, start_acc});
        mem_desc_Vj.init(
            args.V_ptr,
            {end_y, end_x, b_stride * args.uB},
            {start_acc, start_x});
      } else if constexpr (kVarlen) {
        // there is an assumption that kBc should equals to the block size, so
        // the startT should always divisible by kBc
        int32_t start_x;
        int32_t end_x;
        if (args.block_tables != nullptr) {
          int32_t block_slot_offset = startT / args.block_size;
          int32_t start_x_offset =
              args.block_tables
                  [batch_id * args.max_blocks_per_seq + block_slot_offset];
          start_x = start_x_offset * args.block_size;
          // end_x = start_x + args.block_size;
          int32_t seqlen = args.cu_seqlen_k[batch_id];
          int32_t remain_T = seqlen - startT;
          remain_T = remain_T < args.block_size ? remain_T : args.block_size;
          end_x = start_x + remain_T;
        } else {
          start_x = startT + kv_offset_y;
          end_x = start_x + kBc;
          int32_t limit_x = kv_offset_y + args.cu_seqlen_k[batch_id];
          end_x = end_x < limit_x ? end_x : limit_x;
        }
        int32_t start_acc = head_id_kv * args.uH;
        uint32_t end_y = start_acc + args.uH;
        mem_desc_Kj_T.init(
            args.K_ptr,
            {end_x, end_y, args.uNkv * args.uH},
            {start_x, start_acc});

        mem_desc_Vj.init(
            args.V_ptr,
            {end_y, end_x, args.uNkv * args.uH},
            {start_acc, start_x});

      } else {
        int32_t start_x = kv_offset_y + startT;
        uint32_t end_x = start_x + kBc;
        uint32_t boundary_x = kv_offset_y + args.uT;
        end_x = end_x > boundary_x ? boundary_x : end_x;

        int32_t start_acc = kv_offset_x;
        uint32_t end_acc = start_acc + args.uH;

        mem_desc_Kj_T.init(
            args.K_ptr,
            {end_x, end_acc, args.kv_strideT},
            {start_x, start_acc});
        mem_desc_Vj.init(
            args.V_ptr,
            {end_acc, end_x, args.kv_strideT},
            {start_acc, start_x});
      }

      // B, N, 1, T
      // gid * T + startT
      if constexpr (kUseAlibi && !kVarlen) {
        int32_t start_x = startT;
        uint32_t end_x = start_x + kBc;
        uint32_t boundary_x = args.uT;
        end_x = end_x > boundary_x ? boundary_x : end_x;

        int32_t start_y = gid;
        uint32_t end_y = start_y + 1;

        mem_desc_Ai.init(
            args.A_ptr, {end_x, end_y, args.uAT}, {start_x, start_y});
      }
    }
  };

  context_t ctx;

  // ======================= // gemm_Sij // ======================= //
  // Define kernel to compute Sij = Qi x Kj.T
  /// @brief gemm_Sij is used to compute Sij = Qi x Kj.T
  /// # [4 *[Br,H]] x [H,Bc] = [Br,Bc]
  inline void gemm_Sij(
      std::array<matAccSij_t, kHeadPerKv>& matAccSij_s,
      arguments_t& args) {
    using gemm_args_t = typename gemm_Sij_t::arguments_t;

    // Gemm to comput Sij
    gemm_Sij_t gemm;
    uint32_t loop_count = (args.uH + accum_step - 1) / accum_step;
    gemm_args_t gemm_args(ctx.mem_desc_Qi_L, ctx.mem_desc_Kj_T, loop_count);
    gemm(ctx.g, matAccSij_s, gemm_args);

    // Multiply by softmax scaling factor
    // bmm * alpha
    for (auto& matAccSij : matAccSij_s) {
      matAccSij.reg *= args.sm_scale;
      if (args.softcap > 0.0) {
        matAccSij.reg /= args.softcap;
        tanh_t tanh;
        tanh(matAccSij, 0);
        matAccSij.reg *= args.softcap;
      }
      // + alibi
      if constexpr (kUseAlibi && !kVarlen) {
        using alibi_op_t = bias_add_op_t<scalar_t, arch_tag>;
        using alibi_args_t = typename alibi_op_t::arguments_t;

        int32_t tile_offset_x = ctx.sg_idx * kSgBc;
        int32_t tile_offset_y = 0;
        ctx.mem_desc_Ai.update_coord(tile_offset_x, tile_offset_y);

        alibi_op_t alibi_op;
        alibi_args_t alibi_args(ctx.mem_desc_Ai.base, ctx.mem_desc_Ai.shape);
        alibi_op(matAccSij, ctx.mem_desc_Ai.coord, alibi_args);
      }

      // Add attn_mask if needed
      if constexpr (kUseBias && !kIsCausal) {
        if (args.is_bias_add) {
          using mask_op_t = bias_add_op_t<scalar_t, arch_tag>;
          using mask_args_t = typename mask_op_t::arguments_t;
          int32_t tile_offset_x = ctx.sg_idx * kSgBc;
          ctx.mem_desc_Bij.update_coord_x(tile_offset_x);
          mask_op_t mask_op;
          mask_args_t mask_args(ctx.mem_desc_Bij.base, ctx.mem_desc_Bij.shape);
          mask_op(matAccSij, ctx.mem_desc_Bij.coord, mask_args);
        } else {
          using mask_op_t = subgroup::
              elemwise_reduce_op_t<reduce_op::sum, scalar_t, arch_tag>;
          using mask_args_t = typename mask_op_t::arguments_t;

          int32_t tile_offset_x = ctx.sg_idx * kSgBc;
          int32_t tile_offset_y = ctx.sg_idy * kSgBr;
          ctx.mem_desc_Bij.update_coord(tile_offset_x, tile_offset_y);

          mask_op_t mask_op;
          mask_args_t mask_args(ctx.mem_desc_Bij.base, ctx.mem_desc_Bij.shape);
          mask_op(matAccSij, ctx.mem_desc_Bij.coord, mask_args);
        }
      }

      // Add attn_mask if needed
      if constexpr (kUseBias && kIsCausal) {
        using bias_op_t = bias_add_op_t<scalar_t, arch_tag>;
        using bias_args_t = typename bias_op_t::arguments_t;

        int32_t tile_offset_x = ctx.sg_idx * kSgBc;
        int32_t tile_offset_y = 0;
        ctx.mem_desc_Bij.update_coord(tile_offset_x, tile_offset_y);

        bias_op_t bias_op;
        bias_args_t bias_args(ctx.mem_desc_Bij.base, ctx.mem_desc_Bij.shape);
        bias_op(matAccSij, ctx.mem_desc_Bij.coord, bias_args);
      }
    }
  }
  // ======================= // gemm_Oi // ======================= //
  // Define kernel to compute Oi += Pij x Vj
  using gemm_Oi_t = group::gqa_t<
      compute_policy_BrBm,
      tile_shape_BrHm,
      mem_desc_Pij_L_t,
      mem_desc_Vj_t,
      kHeadPerKv>;
  using matAccOi_t = typename gemm_Oi_t::matAcc_t;
  using matOi_t = subgroup::tile_t<
      scalar_t,
      subgroup::tile_desc_t<
          matAccOi_t::tile_size_x,
          matAccOi_t::tile_size_y,
          matAccOi_t::block_size_x,
          matAccOi_t::block_size_y,
          reg_layout::tiled>>;

  /// @brief gemm_Oi is used to compute Oi += Pij x Vj
  /// # [Br,Bc] x [Bc,H] = [Br,Hm]
  inline void gemm_Oi(
      std::array<matAccOi_t, kHeadPerKv>& matAccOi_s,
      arguments_t& args,
      uint32_t startT) {
    using gemm_args_t = typename gemm_Oi_t::arguments_t;

    uint32_t remainT = args.uT - startT;
    uint32_t boundary_k = remainT > kBc ? kBc : remainT;
    uint32_t loop_count = (boundary_k + accum_step - 1) / accum_step;

    // Gemm to comput Oi
    gemm_Oi_t gemm;
    gemm_args_t gemm_args(ctx.mem_desc_Pij_L, ctx.mem_desc_Vj, loop_count);
    gemm(ctx.g, matAccOi_s, gemm_args);
  }

  // ====================== // apply_mask // ====================== //

  /// @brief apply mask to matAccSij.
  inline void apply_mask(
      matAccSij_t& matAccSij,
      arguments_t& args,
      uint32_t startF,
      uint32_t startT,
      uint32_t iter) {
    using tile_mask = tile_mask_t<matAccSij_t>;

    uint32_t sg_startT = startT + ctx.sg_idx * kSgBc;
    // alibi_slopes: B, N or N
    if constexpr (kUseAlibi && kVarlen) {
      float _alibi_slopes = reinterpret_cast<float*>(args.A_ptr)
          [ctx.batch_id * args.uAT + (ctx.head_id_kv * kHeadPerKv + iter)];
      uint32_t sg_startF = startF + ctx.sg_idy * kSgBr;
      tile_mask::alibi_mask(matAccSij, _alibi_slopes, sg_startT, sg_startF);
    }
    uint32_t real_T = args.uT;
    if constexpr (kVarlen) {
      real_T = args.cu_seqlen_k[ctx.batch_id];
    }
    uint32_t remainT = std::max(int(real_T) - int(sg_startT), 0);
    if constexpr (kIsLocal) {
      uint32_t sg_startF = startF + ctx.sg_idy * kSgBr;
      tile_mask::local_mask(
          matAccSij, sg_startF, sg_startT, args.w_left, args.w_right);
    }
    if (remainT < kSgBc) {
      tile_mask::padding_mask(matAccSij, remainT);
    }

    if constexpr (kIsCausal && !kIsLocal) {
      uint32_t sg_startF = startF + ctx.sg_idy * kSgBr;
      if (sg_startT + kSgBc > sg_startF) {
        tile_mask::causal_mask(matAccSij, sg_startT, sg_startF);
      }
    }
  }
  // ====================== // softmax_fwd // ===================== //

  /// @brief softmax_fwd is used to do softmax.
  inline void softmax_fwd(
      matAccSij_t& matAccSij,
      matAccOi_t& matAccOi,
      [[maybe_unused]] arguments_t& args,
      uint32_t iter) {
    using wg_row_max_t =
        group_row_reduce_t<matAccSij_t, wg_size_x, reduce_op::max, arch_tag>;
    using wg_row_sum_t =
        group_row_reduce_t<matAccSij_t, wg_size_x, reduce_op::sum, arch_tag>;

    // init slm address for group reducer
    uint32_t reducer_slm = softmax_slm +
        (iter * wg_size + ctx.sg_idy * wg_size_x) * kSgBr * sizeof(accum_t);

    // compute new m
    wg_row_max_t wg_row_max(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    xetla_vector<accum_t, kSgBr> m_new = wg_row_max(matAccSij);
    m_new = xetla_max<accum_t, kSgBr>(m_new, ctx.softmax_m[iter]);

    xetla_mask<kSgBr> mask_inf = m_new == kNegInfinity;
    m_new.xetla_merge(0.f, mask_inf);

    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive();

    // correct old l
    ctx.softmax_l[iter] *=
        xetla_exp<accum_t, kSgBr>(ctx.softmax_m[iter] - m_new);
    // compute Pij
    subgroup::tile_broadcast_op<subgroup::tile_minus, matAccSij_t>(
        matAccSij, m_new);
    // matAccSij.reg = xetla_exp<accum_t>(matAccSij.reg);

    matAccSij_t mat_zeros(0);
    constexpr int elems = matAccSij_t::tile_desc::tile_elems;
    // xetla_mask<elems> mask = matAccSij->reg < (INFINITY * -1) ||
    // matAccSij->reg > INFINITY;
    xetla_mask<elems> mask = matAccSij.reg < -65400.f;
    (matAccSij.reg)
        .xetla_merge(mat_zeros.reg, xetla_exp<accum_t>(matAccSij.reg), mask);

    if constexpr (wg_size_x > 1)
      ctx.nbarrier.wait();

    // compute new l
    wg_row_sum_t wg_row_sum(ctx.sg_idx, ctx.sg_idy, reducer_slm);
    xetla_vector<accum_t, kSgBr> l_new = wg_row_sum(matAccSij);
    l_new += ctx.softmax_l[iter];

    // rescale operands of matmuls
    xetla_vector<accum_t, kSgBr> o_scale =
        xetla_exp<accum_t, kSgBr>(ctx.softmax_m[iter] - m_new);
    subgroup::tile_broadcast_op<subgroup::tile_mul, matAccOi_t>(
        matAccOi, o_scale);
    // update m and l for the next step
    ctx.softmax_m[iter] = m_new;
    ctx.softmax_l[iter] = l_new;

    // save Pij to local memory
    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<arch_tag>,
        tile_shape_BrBc,
        mem_desc_Pij_L_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, matAccSij, ctx.mem_desc_Pij_L[iter]);
    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
  }

  // ==================== // store_logsumsoftmax // ====================== //

  /// @brief store logsumsoftmax to global memory. [B,N,F]
  inline void store_for_backward([[maybe_unused]] const arguments_t& args) {
    // save m and l to global
    if constexpr (!kIsTraining) {
      return;
    }
  }
  // == == == == == == == == == == // raw_store_Oi // ====================== //

  /// @brief store raw Oi to global memory. [B,F,N,H]
  inline void rescale_then_store_Oi(
      matAccOi_t& matAccOi,
      [[maybe_unused]] arguments_t& args,
      uint32_t iter) {
    scalar_t sink = args.S_ptr != nullptr
        ? reinterpret_cast<scalar_t*>(
              args.S_ptr)[ctx.head_id_start * kHeadPerKv + iter]
        : static_cast<scalar_t>(kNegInfinity);
    ctx.softmax_l[iter] += xetla_exp<accum_t, kSgBr>(
        static_cast<accum_t>(sink) - ctx.softmax_m[iter]);
    subgroup::tile_broadcast_op<subgroup::tile_mul, matAccOi_t>(
        matAccOi, 1 / ctx.softmax_l[iter]);
    using epilogue_t = group::epilogue_t<
        group::epilogue_policy_default<arch_tag>,
        tile_shape_BrHm,
        mem_desc_Oi_t>;
    epilogue_t epilogue;
    epilogue(ctx.g, matAccOi, ctx.mem_desc_Oi[iter]);
  }

  // ====================== // preload_Qi // ====================== //

  /// @brief preload_Qi is used to load Qi from global to local memory.
  inline void preload_Qi(
      [[maybe_unused]] arguments_t& args,
      uint32_t iter = 0) {
    using matQi_tile_desc_t = typename gemm_Oi_t::matAcc_tile_desc_t;
    using matQi_t = subgroup::tile_t<scalar_t, matQi_tile_desc_t>;
    using matQi_load_t = subgroup::mem_payload_t<
        mem_mask_desc_t<scalar_t, mem_desc_Qi_t::layout, mem_desc_Qi_t::space>,
        matQi_tile_desc_t,
        subgroup::msg_type_v<
            matQi_tile_desc_t,
            mem_mask_desc_t<
                scalar_t,
                mem_desc_Qi_t::layout,
                mem_desc_Qi_t::space>>,
        arch_tag>;
    using matQi_store_t = subgroup::mem_payload_t<
        mem_mask_desc_t<
            scalar_t,
            mem_desc_Qi_L_t::layout,
            mem_desc_Qi_L_t::space>,
        matQi_tile_desc_t,
        subgroup::msg_type_v<
            matQi_tile_desc_t,
            mem_mask_desc_t<
                scalar_t,
                mem_desc_Qi_L_t::layout,
                mem_desc_Qi_L_t::space>>,
        arch_tag>;

    int32_t tile_offset_x = ctx.sg_idx * kSgHm;
    int32_t tile_offset_y = ctx.sg_idy * kSgBr;

    mem_desc_Qi_t mem_desc_Qi_load(ctx.mem_desc_Qi[iter]);
    mem_desc_Qi_L_t mem_desc_Qi_store(ctx.mem_desc_Qi_L[iter]);

    mem_desc_Qi_load.update_coord(tile_offset_x, tile_offset_y);
    mem_desc_Qi_store.update_coord(tile_offset_x, tile_offset_y);

    matQi_t matQi;
    matQi_load_t matQi_load(mem_desc_Qi_load);
    subgroup::tile_load(matQi, matQi_load);

    matQi_store_t matQi_store(mem_desc_Qi_store);
    subgroup::tile_store(matQi, matQi_store);

    xetla_fence<memory_kind::shared_local>();
    if constexpr (wg_size_x > 1)
      ctx.nbarrier.arrive_wait();
  }

 public:
  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  inline static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t barrier_count_Sij = gemm_Sij_t::barrier_count;
    constexpr uint32_t barrier_count_Oi = gemm_Oi_t::barrier_count;
    constexpr uint32_t count = std::max(barrier_count_Sij, barrier_count_Oi);
    static_assert(
        count <= 32, "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = slm_size_Qi + slm_size_Pij + slm_size_softmax;
    static_assert(
        size <= (arch_attr_t<arch_tag>::local_mem_size),
        "The local memory size should be less than arch total local memory size");
    return size;
  };

  inline static void check_slm_size(const sycl::device& d) {
    constexpr auto slm_size = get_slm_size();
    if (slm_size > d.get_info<sycl::info::device::local_mem_size>())
      throw std::runtime_error(
          "Head SLM size too large for the current device!");
  }

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static inline sycl::nd_range<3> get_nd_range(
      uint32_t total_batches,
      uint32_t num_queries) {
    // local range
    static const sycl::range<3> local_range =
        sycl::range<3>{1, wg_size_y, wg_size_x};
    // group range
    uint32_t group_range_m = (num_queries + kBr - 1) / kBr;
    sycl::range<3> group_range{total_batches, group_range_m, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };
  // ================= // Entry of the functor // ================= //

  inline KERNEL_FUNC void operator()(nd_item<3>& item, arguments_t& args) {
    // allocate slm and nbarrier resource
    xetla_local_init<get_slm_size()>();
    xetla_nbarrier_init<get_barrier_count()>();

    // initialize context for flash mha loops
    ctx.init_context(item, args);
    uint32_t batch_id = ctx.batch_id; // get batch idx
    // Early exit when current thread access data exceed actual seqlen in varlen
    // fwd
    int32_t actual_seqlen_q = args.uF;
    if constexpr (kVarlen) {
      actual_seqlen_q =
          args.cu_seqlen_q[batch_id + 1] - args.cu_seqlen_q[batch_id];
      int32_t seqlen_q = item.get_group(1) * kBr;

      if (seqlen_q >= actual_seqlen_q) {
        return;
      }
    }
// preload Qi to local memory
#pragma unroll
    for (uint32_t i = 0; i < kHeadPerKv; i++) {
      preload_Qi(args, i);
    }
    // initialize matAccOi for accumulate the output
    std::array<matAccOi_t, kHeadPerKv> matAccOi_s = {0};

    uint32_t startF = item.get_group(1) /* 0 */ * kBr /* 64 */;
    uint32_t endF = std::min(startF + kBr, args.uF);
    int32_t actual_seqlen_k = 0;
    int32_t seqlen_diff = 0;
    if constexpr (kVarlen) {
      actual_seqlen_k = args.cu_seqlen_k[batch_id];
      seqlen_diff = actual_seqlen_k - actual_seqlen_q;
    }

    // iterate through the keys
    for (uint32_t startT = 0; startT < args.uT; startT += kBc) {
      // Early leave for varlen_fwd if we found current seqlen exceed the actual
      // seqlen.
      if constexpr (kVarlen) {
        if (startT >= actual_seqlen_k) {
          break;
        }
        if constexpr (kIsLocal) {
          if (startT + kBc <= ctx.local_left) {
            continue;
          }
          if (startT > ctx.local_right) {
            break;
          }
        }
      }
      if constexpr (kIsCausal) {
        if (startT >= endF + seqlen_diff)
          break;
      }
      // update context for current loop
      ctx.update_context(args, startT);
      // compute Sij
      std::array<matAccSij_t, kHeadPerKv> matAccSij_s = {0};
      gemm_Sij(matAccSij_s, args);
#pragma unroll
      for (uint32_t i = 0; i < kHeadPerKv; i++) {
        auto& matAccSij = matAccSij_s[i];
        auto& matAccOi = matAccOi_s[i];
        // apply mask
        apply_mask(matAccSij, args, startF + seqlen_diff, startT, i);
        // softmax
        softmax_fwd(matAccSij, matAccOi, args, i);
      }
      // compute Oi
      gemm_Oi(matAccOi_s, args, startT);
    }

// Store output to global
#pragma unroll
    for (uint32_t i = 0; i < kHeadPerKv; i++) {
      auto& matAccOi = matAccOi_s[i];
      // rescale and store Oi
      rescale_then_store_Oi(matAccOi, args, i);
    }
    store_for_backward(args);
  }
}; // fmha_forward_t
} // namespace fmha
} // namespace gpu::xetla
