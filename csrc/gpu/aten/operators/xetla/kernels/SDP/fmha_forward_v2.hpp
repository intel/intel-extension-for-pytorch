
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
    bool kUseBias>
class fmha_forward_v2_t {
 public:
  using accum_t = float;

  struct arguments_t {
    const scaler_t* query;
    const scaler_t* key;
    const scaler_t* value;
    const scaler_t* mask;
    scaler_t* output;

    const size_t num_batch;
    const size_t num_heads;
    const size_t num_kv_heads;
    const size_t ctx_len;

    accum_t sm_scale;

    const size_t q_batch_step;
    const size_t q_head_step;
    const size_t k_batch_step;
    const size_t k_head_step;
    const size_t k_seq_step;
    const size_t v_batch_step;
    const size_t v_head_step;
    const size_t v_seq_step;
    const size_t mask_batch_step;
    const size_t mask_head_step;
    const size_t out_batch_step;
    const size_t out_head_step;
  };
  static constexpr size_t LOAD_BYTES_LEN = 128;
  static constexpr size_t O_offset = 0;
  static constexpr size_t O_size = sg_num * head_dim * sizeof(scaler_t);
  static constexpr size_t L_offset = O_size;
  static constexpr size_t L_size = sg_num * sizeof(accum_t);
  static constexpr size_t M_offset = L_offset + L_size;
  static constexpr size_t M_size = sg_num * sizeof(accum_t);
  xetla_nbarrier_t<sg_num, sg_num, arch_tag> nbarrier;

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
    const size_t kv_group_num = args.num_heads / args.num_kv_heads;
    const size_t sg_ctx_len = args.ctx_len / sg_num;
    const size_t rem_ctx_len = args.ctx_len % sg_num;
    const size_t sg_head_dim = head_dim / sg_num;
    static_assert(head_dim % sg_num == 0);

    const size_t batch_idx = item.get_group(0);
    const size_t head_idx = item.get_group(1);
    const size_t kv_head_idx = head_idx / kv_group_num;
    const size_t sg_id = item.get_local_id(2);
    xetla_local_init<get_slm_size()>();
    nbarrier.init_nbarrier(sg_id, nbarrier_role::producer_consumer);

    const auto query_head = args.query + batch_idx * args.q_batch_step +
        head_idx * args.q_head_step;
    const auto key_head = args.key + batch_idx * args.k_batch_step +
        kv_head_idx * args.k_head_step;
    const auto value_head = args.value + batch_idx * args.v_batch_step +
        kv_head_idx * args.v_head_step;
    const auto mask_head = args.mask + batch_idx * args.mask_batch_step +
        head_idx * args.mask_head_step;
    const auto output_head = args.output + batch_idx * args.out_batch_step +
        head_idx * args.out_head_step;

    xetla_vector<fp16, head_dim> q;
    constexpr int BLK = LOAD_BYTES_LEN / sizeof(scaler_t);
    static_assert(head_dim % BLK == 0);
#pragma unroll
    for (int i = 0; i < head_dim; i += BLK) {
      q.xetla_select<BLK, 1>(i) =
          xetla_load_global<scaler_t, BLK>(query_head, sizeof(scaler_t) * i);
    }

    size_t start_ctx_id = sg_ctx_len * sg_id + std::min(sg_id, rem_ctx_len);
    size_t end_ctx_id =
        start_ctx_id + sg_ctx_len + (sg_id < rem_ctx_len ? 1 : 0);

    // M_i = max(S_i)
    accum_t M = std::numeric_limits<accum_t>::lowest();
    // L_i = sum(exp(S_i) - M_i)
    accum_t L = std::numeric_limits<accum_t>::min();
    xetla_vector<accum_t, head_dim> O = 0;
    for (size_t i = start_ctx_id; i < end_ctx_id; ++i) {
      const auto k_i = xetla_load_global<scaler_t, head_dim>(
          key_head, sizeof(scaler_t) * i * args.k_seq_step);
      accum_t attn =
          xetla_reduce<accum_t, accum_t, head_dim, reduce_op::sum>(q * k_i);
      accum_t S;
      if constexpr (kUseBias) {
        S = attn * args.sm_scale + mask_head[i];
      } else {
        S = attn * args.sm_scale;
      }
      accum_t M_old = M;
      M = std::max(M, S);
      accum_t attn_exp = xetla_exp(S - M);
      accum_t L_old = L * xetla_exp(M_old - M);
      L = L_old + attn_exp;
      const auto v_i = xetla_load_global<scaler_t, head_dim>(
          value_head, sizeof(scaler_t) * i * args.v_seq_step);
      O = (O * L_old + v_i * attn_exp) / L;
    }

#pragma unroll
    for (int i = 0; i < head_dim; i += BLK) {
      xetla_vector<uint32_t, BLK> offset_i(
          O_offset + (sg_id + i * sg_num) * sizeof(scaler_t),
          sg_num * sizeof(scaler_t));
      xetla_vector<scaler_t, BLK> O_i =
          xetla_cvt<scaler_t, accum_t, BLK>(O.xetla_select<BLK, 1>(i));
      xetla_store_local<scaler_t, 1, data_size::default_size, BLK>(
          offset_i, O_i);
    }

    xetla_store_local<accum_t, 1>(L_offset + sg_id * sizeof(accum_t), L);
    xetla_store_local<accum_t, 1>(M_offset + sg_id * sizeof(accum_t), M);

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    xetla_vector<accum_t, sg_num> M_sg =
        xetla_load_local<accum_t, sg_num>(M_offset);
    accum_t M_total =
        xetla_reduce<accum_t, accum_t, sg_num, reduce_op::max>(M_sg);

    xetla_vector<accum_t, sg_num> L_sg =
        xetla_load_local<accum_t, sg_num>(L_offset);
    L_sg *= xetla_exp<accum_t, sg_num>(M_sg - M_total);
    accum_t L_total =
        xetla_reduce<accum_t, accum_t, sg_num, reduce_op::sum>(L_sg);
    xetla_vector<accum_t, sg_num> L_ratio = L_sg / L_total;
    const size_t start_idx = sg_head_dim * sg_id;
#pragma unroll
    for (size_t i = start_idx; i < start_idx + sg_head_dim; ++i) {
      accum_t O_total = xetla_reduce<accum_t, accum_t, sg_num, reduce_op::sum>(
          xetla_load_local<scaler_t, sg_num>(
              O_offset + i * sg_num * sizeof(scaler_t)) *
          L_ratio

      );
      output_head[i] = O_total;
    }
  }
}; // fmha_forward_t
} // namespace fmha
} // namespace gpu::xetla
