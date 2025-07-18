
#pragma once
/*
Fused Multi-Head Attention Forward

This is an implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf)
*/
#include "../xetla.h"

namespace gpu::xetla {
namespace fmha {
// Current it handles only next-token inference cases where qlen equals 1
template <
    typename scaler_t,
    gpu_arch arch_tag,
    int sg_num,
    int head_dim,
    uint32_t kHeadPerKv>
class fmha_forward_v4_t {
 public:
  using accum_t = float;

  struct arguments_t {
    const scaler_t* query;
    const scaler_t* key;
    const scaler_t* value;
    scaler_t* output;

    const size_t num_batch;
    const size_t num_heads;
    const size_t num_kv_heads;
    const size_t num_keys;

    accum_t sm_scale;

    const size_t q_seq_step;
    const size_t q_head_step;
    const size_t k_seq_step;
    const size_t k_head_step;
    const size_t v_seq_step;
    const size_t v_head_step;
    const size_t out_seq_step;
    const size_t out_head_step;

    // Sequence length info
    int32_t* cu_seqlen_q;
    int32_t* cu_seqlen_k;

    accum_t softcap;
    int32_t* block_tables; // [B, max_blocks_per_seq]
    uint32_t max_blocks_per_seq;
    uint32_t block_size;
  };
  static constexpr size_t LOAD_BYTES_LEN = 128;
  static constexpr size_t O_offset = 0;
  static constexpr size_t O_size =
      kHeadPerKv * sg_num * head_dim * sizeof(accum_t);
  static constexpr size_t L_offset = O_size;
  static constexpr size_t L_size = kHeadPerKv * sg_num * sizeof(accum_t);
  static constexpr size_t M_offset = L_offset + L_size;
  static constexpr size_t M_size = kHeadPerKv * sg_num * sizeof(accum_t);
  xetla_nbarrier_t<sg_num, sg_num, gpu_arch::XeLpg> nbarrier;

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    const auto size = O_size + L_size + M_size;
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
      uint32_t num_batch,
      uint32_t num_heads) {
    static const sycl::range<3> local_range = sycl::range<3>{1, 1, sg_num};
    sycl::range<3> group_range{num_batch, num_heads, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  inline KERNEL_FUNC void operator()(nd_item<3>& item, arguments_t& args) {
    const size_t sg_head_dim = head_dim / sg_num;
    static_assert(head_dim % sg_num == 0);

    const size_t batch_idx = item.get_group(0);
    const size_t kv_head_idx = item.get_group(1);
    const size_t head_idx_start = kv_head_idx * kHeadPerKv;
    const size_t sg_id = item.get_local_id(2);

    const size_t cur_seq_k =
        args.cu_seqlen_k[batch_idx + 1] - args.cu_seqlen_k[batch_idx];
    const size_t cur_block_k =
        (cur_seq_k + args.block_size - 1) / args.block_size;
    const size_t rem_block_k = cur_seq_k - (cur_block_k - 1) * args.block_size;
    const size_t sg_ctx_len = cur_block_k / sg_num;
    const size_t rem_ctx_len = cur_block_k % sg_num;
    xetla_local_init<get_slm_size()>();
    nbarrier.init_nbarrier(sg_id, nbarrier_role::producer_consumer);

    const auto query_head = args.query +
        args.cu_seqlen_q[batch_idx] * args.q_seq_step +
        head_idx_start * args.q_head_step;
    const auto key_head = args.key;
    const auto value_head = args.value;
    const auto output_head = args.output +
        args.cu_seqlen_q[batch_idx] * args.out_seq_step +
        head_idx_start * args.out_head_step;

    std::array<xetla_vector<fp16, head_dim>, kHeadPerKv> q_s;
    constexpr int BLK = LOAD_BYTES_LEN / sizeof(scaler_t);
    static_assert(head_dim % BLK == 0);
#pragma unroll
    for (uint32_t j = 0; j < kHeadPerKv; j++) {
      auto query_head_j = query_head + j * args.q_head_step;
#pragma unroll
      for (int i = 0; i < head_dim; i += BLK) {
        q_s[j].xetla_select<BLK, 1>(i) = xetla_load_global<scaler_t, BLK>(
            query_head_j, sizeof(scaler_t) * i);
      }
    }
    size_t start_block_k = batch_idx * args.max_blocks_per_seq +
        sg_id * sg_ctx_len + std::min(sg_id, rem_ctx_len);
    size_t end_block_k =
        start_block_k + sg_ctx_len + (sg_id < rem_ctx_len ? 1 : 0);

    // M_i = max(S_i)
    std::array<accum_t, kHeadPerKv> M_s = {
        std::numeric_limits<accum_t>::lowest()};
    // L_i = sum(exp(S_i) - M_i)
    std::array<accum_t, kHeadPerKv> L_s = {std::numeric_limits<accum_t>::min()};
    std::array<xetla_vector<accum_t, head_dim>, kHeadPerKv> O_s = {0};
    for (size_t b_i = start_block_k; b_i < end_block_k; ++b_i) {
      size_t start_i = args.block_tables[b_i] * args.block_size;
      size_t end_i = b_i - start_block_k == cur_block_k - 1
          ? start_i + rem_block_k
          : start_i + args.block_size;
      for (size_t i = start_i; i < end_i; ++i) {
        const auto k_i = xetla_cvt<accum_t, scaler_t, head_dim>(
            xetla_load_global<scaler_t, head_dim>(
                key_head,
                sizeof(scaler_t) *
                    (i * args.k_seq_step + kv_head_idx * args.k_head_step)));
        const auto v_i = xetla_cvt<accum_t, scaler_t, head_dim>(
            xetla_load_global<scaler_t, head_dim>(
                value_head,
                sizeof(scaler_t) *
                    (i * args.v_seq_step + kv_head_idx * args.v_head_step)));
#pragma unroll
        for (uint32_t j = 0; j < kHeadPerKv; ++j) {
          accum_t attn =
              xetla_reduce<accum_t, accum_t, head_dim, reduce_op::sum>(
                  xetla_cvt<accum_t, scaler_t, head_dim>(q_s[j]) * k_i);
          accum_t S;
          S = attn * args.sm_scale;
          accum_t M_old = M_s[j];
          M_s[j] = std::max(M_s[j], S);
          accum_t attn_exp = xetla_exp(S - M_s[j]);
          accum_t L_old = L_s[j] * xetla_exp(M_old - M_s[j]);
          L_s[j] = L_old + attn_exp;
          O_s[j] = (O_s[j] * L_old + v_i * attn_exp) / L_s[j];
        }
      }
    }

#pragma unroll
    for (int j = 0; j < kHeadPerKv; j++) {
#pragma unroll
      for (int i = 0; i < head_dim; i += BLK) {
        xetla_vector<uint32_t, BLK> offset_i(
            O_offset +
                (j * sg_num * head_dim + sg_id + i * sg_num) * sizeof(accum_t),
            sg_num * sizeof(accum_t));
        xetla_vector<accum_t, BLK> O_i = O_s[j].xetla_select<BLK, 1>(i);
        xetla_store_local<accum_t, 1, data_size::default_size, BLK>(
            offset_i, O_i);
      }
    }

#pragma unroll
    for (int j = 0; j < kHeadPerKv; j++) {
      xetla_store_local<accum_t, 1>(
          L_offset + (j * sg_num + sg_id) * sizeof(accum_t), L_s[j]);
      xetla_store_local<accum_t, 1>(
          M_offset + (j * sg_num + sg_id) * sizeof(accum_t), M_s[j]);
    }

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    std::array<xetla_vector<accum_t, sg_num>, kHeadPerKv> M_sg_s;
    std::array<xetla_vector<accum_t, sg_num>, kHeadPerKv> L_sg_s;
    std::array<accum_t, kHeadPerKv> M_total_s;
    std::array<accum_t, kHeadPerKv> L_total_s;
    std::array<xetla_vector<accum_t, sg_num>, kHeadPerKv> L_ratio_s;

#pragma unroll
    for (uint32_t j = 0; j < kHeadPerKv; ++j) {
      M_sg_s[j] = xetla_load_local<accum_t, sg_num>(
          M_offset + (j * sg_num) * sizeof(accum_t));
      M_total_s[j] =
          xetla_reduce<accum_t, accum_t, sg_num, reduce_op::max>(M_sg_s[j]);
      L_sg_s[j] = xetla_load_local<accum_t, sg_num>(
          L_offset + (j * sg_num) * sizeof(accum_t));
      L_sg_s[j] *= xetla_exp<accum_t, sg_num>(M_sg_s[j] - M_total_s[j]);
      L_total_s[j] =
          xetla_reduce<accum_t, accum_t, sg_num, reduce_op::sum>(L_sg_s[j]);
      L_ratio_s[j] = L_sg_s[j] / L_total_s[j];
    }
    const size_t start_idx = sg_head_dim * sg_id;
    std::array<accum_t, kHeadPerKv> O_total_s = {0};
#pragma unroll
    for (size_t j = 0; j < kHeadPerKv; ++j) {
      auto L_ratio = L_ratio_s[j];
      auto output_head_j = output_head + j * args.out_head_step;
#pragma unroll
      for (size_t i = start_idx; i < start_idx + sg_head_dim; ++i) {
        O_total_s[j] = xetla_reduce<accum_t, accum_t, sg_num, reduce_op::sum>(
            xetla_load_local<accum_t, sg_num>(
                O_offset + (j * head_dim + i) * sg_num * sizeof(accum_t)) *
            L_ratio);
        output_head_j[i] = static_cast<scaler_t>(O_total_s[j]);
      }
    }
  }
}; // fmha_forward_t
} // namespace fmha
} // namespace gpu::xetla
