/// Fused Multi-Head Attention
#pragma once

#include <CL/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "fmha_policy.hpp"
#include "kernel_apis.hpp"

namespace esimd {

template <class Policy, class T>
class FusedMha {
 public:
  //
  // Types and Contants
  //
  using Accum = float;
  using MaskT = uint8_t;
  // using ArchConfig = ArchConfig<Policy::kArch>;

  // negative infinity
  static constexpr Accum kNegInfinity = -INFINITY;

  static constexpr uint32_t kHeadSize = Policy::kHeadSize;
  static constexpr uint32_t kNumHeadsQO = Policy::kNumHeadsQO;
  static constexpr uint32_t kNumHeadsKV = Policy::kNumHeadsKV;
  static constexpr uint32_t kSgSeqQO = Policy::kSgSeqQO;
  static constexpr uint32_t kSgSeqKV = Policy::kSgSeqKV;
  static constexpr uint32_t kNumSgX = Policy::kNumSgX;
  static constexpr uint32_t kNumSgY = Policy::kNumSgY;
  static constexpr uint32_t kSimd = Policy::kSimd;
  static constexpr bool kHeadFirst = Policy::kHeadFirst;
  static constexpr bool kUseMask = Policy::kUseMask;

  static constexpr uint32_t kSeqStrideQO =
      kHeadFirst ? kHeadSize : kNumHeadsQO * kHeadSize;
  static constexpr uint32_t kSeqStrideKV =
      kHeadFirst ? kHeadSize : kNumHeadsKV * kHeadSize;

  static constexpr uint32_t kWgSeqQO = kSgSeqQO * kNumSgY;
  static constexpr uint32_t kWgSeqKV = kSgSeqKV * kNumSgX;

  static constexpr uint32_t kMaxLoadLen = ArchConfig::kMaxLoadBytes /
              sizeof(T) >
          kHeadSize
      ? kHeadSize
      : ArchConfig::kMaxLoadBytes / sizeof(T);
  static constexpr uint32_t kLoadLen = get_max_power_of_2<kMaxLoadLen>();
  static constexpr uint32_t kRemainLoadLen = kHeadSize % kLoadLen;
  static constexpr uint32_t kRemainHeadSimd = kHeadSize % kSimd;

  static_assert(
      (kSgSeqKV & (kSgSeqKV - 1)) == 0,
      "kSgSeqKV should be power of 2");
  static_assert((kNumSgX & (kNumSgX - 1)) == 0, "kNumSgX should be power of 2");
  static_assert((kNumSgY & (kNumSgY - 1)) == 0, "kNumSgY should be power of 2");
  static_assert(
      kNumHeadsQO % kNumHeadsKV == 0,
      "kNumHeadsQO must be a multiple of kNumHeadsKV");
  static_assert(
      kHeadSize % kSimd == 0,
      "Head size must be a multiple of kSimd");

  //
  // Constructor
  //
  inline FusedMha()
      : cache_q_(0.0f),
        accum_o_(0.0f),
        softmax_m_(kNegInfinity),
        softmax_s_(0.0f),
        mask_qk_(1) {}

  //
  // Arguments for the functor
  //
  struct Arguments {
    T* query;
    T* key;
    T* value;
    T* out;
    MaskT* mask; // (batch_size, num_head, seq_len_q, seq_ken_kv)
    uint32_t qo_len;
    uint32_t kv_len;
    // softmax scale
    Accum sfm_scale;

    inline Arguments(
        T* query_,
        T* key_,
        T* value_,
        T* out_,
        MaskT* mask_,
        uint32_t qo_len_,
        uint32_t kv_len_,
        Accum sfm_scale_)
        : query(query_),
          key(key_),
          value(value_),
          out(out_),
          mask(mask_),
          qo_len(qo_len_),
          kv_len(kv_len_),
          sfm_scale(sfm_scale_) {}
  };

  //
  // Return the required nd_range
  //
  inline static auto nd_range(uint32_t num_batches, uint32_t qo_len) {
    uint32_t group_range_y = (qo_len + kWgSeqQO - 1) / kWgSeqQO;

    auto local_range = sycl::range<2>{kNumSgY, kNumSgX};
    auto group_range = sycl::range<2>{num_batches * kNumHeadsQO, group_range_y};

    return sycl::nd_range<2>{group_range * local_range, local_range};
  }

  // SLM size for temporary out storage
  static constexpr uint32_t kSlmOutSize =
      kWgSeqQO * kHeadSize * kNumSgX * sizeof(Accum);
  static constexpr uint32_t kSlmSoftmax = kWgSeqQO * kNumSgX * sizeof(Accum);
  static constexpr uint32_t kSlmSize =
      kSlmSoftmax < kSlmOutSize ? kSlmOutSize : kSlmSoftmax;

  static constexpr uint32_t slm_out_ptr = 0;
  static constexpr uint32_t slm_softmax_ptr = 0;

  //
  // Return the required shared local memory size
  //
  inline static constexpr uint32_t slm_size() {
    static_assert(
        kSlmSize <= (64 * 1024),
        "The local memory size should be less than 64KB!");
    return kSlmSize;
  }

  //
  // Entry point
  //
  inline SYCL_ESIMD_FUNCTION void operator()(
      sycl::nd_item<2>& ndi,
      Arguments& args) {
    init(ndi, args);
    load_query(args);

    prefetch_kv(args);
    update_prefetch(args);

    for (uint32_t i = 0; i < (args.kv_len + kWgSeqKV - 1) / kWgSeqKV; ++i) {
      load_mask(args);
      prefetch_kv(args);
      gemm_qk(ndi, args);
      apply_mask<kSgSeqQO>(args);
      softmax(ndi, args);
      gemm_qkv(ndi, args);
      update(args);
      update_prefetch(args);
    }

    process_out(ndi, args);
  }

 private:
  Vec<Accum, kSgSeqQO * kHeadSize> cache_q_; // cached query
  Vec<Accum, kSgSeqQO * kSgSeqKV> sfm_qk_; // query x key for softmax
  Vec<Accum, kSgSeqQO * kHeadSize> accum_o_; // accumulated out
  Vec<Accum, kSgSeqQO> softmax_m_;
  Vec<Accum, kSgSeqQO> softmax_s_;
  Vec<MaskT, kSgSeqQO * kSgSeqKV> mask_qk_;
  uint16_t mask_qk_valid_;

  uint32_t sg_idx_;
  uint32_t sg_idy_;
  uint32_t offset_qo_; // pointer offset of query (out)
  uint32_t offset_kv_; // pointer offset of key (value)
  uint32_t seq_start_qo_; // sequence offset of query (out)
  uint32_t seq_start_kv_; // sequence offset of key (value)
  uint64_t mask_start_; // sequence offset of mask
  uint32_t prefetch_seq_start_kv_; // sequence offset of key (value)
  uint32_t slm_offset_load_; // load offset of SLM for group reduce
  uint32_t slm_offset_store_; // store offset of SLM for group reduce

  inline void init(sycl::nd_item<2>& ndi, Arguments& args) {
    __ESIMD_NS::slm_init(slm_size());

    sg_idx_ = ndi.get_local_id(1);
    sg_idy_ = ndi.get_local_id(0);

    uint32_t group_seq_id = ndi.get_group(1);
    uint32_t group_id = ndi.get_group(0);

    uint32_t batch_id = group_id / kNumHeadsQO;
    uint32_t head_id_q = group_id % kNumHeadsQO;
    uint32_t head_id_kv = head_id_q / (kNumHeadsQO / kNumHeadsKV);

    seq_start_qo_ = group_seq_id * kWgSeqQO + sg_idy_ * kSgSeqQO;
    seq_start_kv_ = sg_idx_ * kSgSeqKV;
    prefetch_seq_start_kv_ = 0;

    if constexpr (kUseMask) {
      mask_start_ =
          (batch_id * kNumHeadsQO + head_id_q) * args.qo_len * args.kv_len;
    }

    if constexpr (kHeadFirst) {
      offset_qo_ =
          (batch_id * kNumHeadsQO + head_id_q) * args.qo_len * kHeadSize;
      offset_kv_ =
          (batch_id * kNumHeadsKV + head_id_kv) * args.kv_len * kHeadSize;
    } else {
      offset_qo_ =
          (batch_id * args.qo_len * kNumHeadsQO + head_id_q) * kHeadSize;
      offset_kv_ =
          (batch_id * args.kv_len * kNumHeadsKV + head_id_kv) * kHeadSize;
    }

    if constexpr (kNumSgX > 1) {
      slm_offset_load_ =
          slm_softmax_ptr + sg_idy_ * kSgSeqQO * kNumSgX * sizeof(Accum);
      slm_offset_store_ = slm_offset_load_ + sg_idx_ * kSgSeqQO * sizeof(Accum);
    }
  }

  inline void update(Arguments& args) {
    seq_start_kv_ += kWgSeqKV;
  }

  inline void update_prefetch(Arguments& args) {
    prefetch_seq_start_kv_ += kWgSeqKV;
  }

  inline void load_query(Arguments& args) {
    // prefetch query
#pragma unroll
    for (uint32_t i = 0; i < kSgSeqQO; ++i) {
      uint32_t seq_offset_qo = seq_start_qo_ + i;
      if (seq_offset_qo < args.qo_len) {
        uint32_t offset = offset_qo_ + seq_offset_qo * kSeqStrideQO;
        cooperative_prefetch_global<T, kHeadSize, kNumSgX>(
            args.query, offset, sg_idx_);
      }
    }
    SW_BARRIER();

    // load kSgSeqQO lines one by one
#pragma unroll
    for (uint32_t i = 0; i < kSgSeqQO; ++i) {
      uint32_t seq_offset_qo = seq_start_qo_ + i;
      if (seq_offset_qo < args.qo_len) {
        uint32_t offset = offset_qo_ + seq_offset_qo * kSeqStrideQO;
#pragma unroll
        for (uint32_t l = 0; l < kHeadSize - kRemainLoadLen; l += kLoadLen) {
          // Load query with length of kLoadLen
          cache_q_.template select<kLoadLen, 1>(i * kHeadSize + l) =
              load_global<T, kLoadLen>(args.query, offset + l);
        }
        if constexpr (kRemainLoadLen > 0) {
          cache_q_.template select<kRemainLoadLen, 1>(
              i * kHeadSize + kHeadSize - kRemainLoadLen) =
              load_global<T, kRemainLoadLen>(
                  args.query, offset + kHeadSize - kRemainLoadLen);
        }
      }
    }
  }
  // TODO: fix me, no exception handling for kv seqlen boundary
  inline void load_mask(Arguments& args) {
    if constexpr (!kUseMask) {
      return;
    }
    uint32_t offset = mask_start_ + seq_start_kv_;
    for (uint32_t i = 0; i < kSgSeqQO; ++i) {
      uint32_t seq_offset_qo = seq_start_qo_ + i;
      if (seq_offset_qo < args.qo_len && seq_start_kv_ < args.kv_len) {
        mask_qk_.template select<kSgSeqKV, 1>(i * kSgSeqKV) =
            load_global_unaligned<MaskT, kSgSeqKV>(
                args.mask, offset + seq_offset_qo * args.kv_len);
      }
    }

    auto reduced_mask_qk =
        col_reduce<BinaryOp::SUM, MaskT, kSgSeqQO, kSgSeqKV>(mask_qk_);
    mask_qk_valid_ = reduced_mask_qk.any();
  }

  // cooperatively prefetch key, value
  // prefetch size: kSgSeqKV * kHeadSize
  inline void prefetch_kv(Arguments& args) {
    // #pragma unroll
    //     for (uint32_t i = 0; i < kSgSeqKV; ++i) {
    //       uint32_t seq_offset_kv = prefetch_seq_start_kv_ + i * kNumSgX +
    //       sg_idx_; if (seq_offset_kv < args.kv_len) {
    //         uint32_t offset = offset_kv_ + seq_offset_kv * kSeqStrideKV;
    //         cooperative_prefetch_global<T, kHeadSize, kNumSgY>(args.key,
    //         offset,
    //                                                            sg_idy_);
    //         cooperative_prefetch_global<T, kHeadSize, kNumSgY>(args.value,
    //         offset,
    //                                                            sg_idy_);
    //       }
    //     }
  }

  //
  // Compute qk = query * key
  //
  inline void gemm_qk(sycl::nd_item<2>& ndi, Arguments& args) {
    if constexpr (kUseMask) {
      if (mask_qk_valid_ == 0) {
        sfm_qk_ = kNegInfinity;
        return;
      }
    }

    // temporary storage for qk
    Vec<Accum, kSgSeqQO* kSgSeqKV* kSimd> tmp_qk = 0.0f;

#pragma unroll
    for (uint32_t j = 0; j < kSgSeqKV; ++j) {
      uint32_t seq_offset_kv = seq_start_kv_ + j;
      if (seq_offset_kv < args.kv_len) {
        uint32_t offset = offset_kv_ + seq_offset_kv * kSeqStrideKV;

        // load key
        Vec<T, kHeadSize> tmp_k;
#pragma unroll
        for (uint32_t l = 0; l < kHeadSize - kRemainLoadLen; l += kLoadLen) {
          // load key with length of kLoadLen
          tmp_k.template select<kLoadLen, 1>(0) =
              load_global<T, kLoadLen>(args.key, offset + l);
        }
        if constexpr (kRemainLoadLen) {
          tmp_k.template select<kRemainLoadLen, 1>(kHeadSize - kRemainLoadLen) =
              load_global<T, kRemainLoadLen>(
                  args.key, offset + kHeadSize - kRemainLoadLen);
        }

        // compute key * value
#pragma unroll
        for (uint32_t i = 0; i < kSgSeqQO; ++i) {
          auto tmp_qk_sub =
              tmp_qk.template select<kSimd, 1>((i * kSgSeqKV + j) * kSimd);
          if constexpr (kUseMask) {
            MaskT is_valid_kv =
                mask_qk_.template select<1, 1>(i * kSgSeqKV + j)[0];
            tmp_qk_sub.merge(kNegInfinity, is_valid_kv == 0);
          }

#pragma unroll
          for (uint32_t s = 0; s < kHeadSize - kRemainHeadSimd; s += kSimd) {
            auto q_sub = cache_q_.template select<kSimd, 1>(i * kHeadSize + s);
            auto k_sub = tmp_k.template select<kSimd, 1>(s);
            tmp_qk_sub += q_sub * k_sub;
          }

          if constexpr (kRemainHeadSimd > 0) {
            auto q_tail = cache_q_.template select<kRemainHeadSimd, 1>(
                i * kHeadSize + kHeadSize - kRemainHeadSimd);
            auto k_tail = tmp_k.template select<kRemainHeadSimd, 1>(
                kHeadSize - kRemainHeadSimd);
            tmp_qk_sub.template select<kRemainHeadSimd, 1>(0) +=
                q_tail * k_tail;
          }
        }
      }
    }
    // reduce tmp_qk to shape of kSgSeqQO * kSgSeqKV
    sfm_qk_ =
        row_reduce<BinaryOp::SUM, Accum, kSgSeqQO * kSgSeqKV, kSimd>(tmp_qk);
  }

  //
  // Apply mask to qk
  //
  template <int N = kSgSeqQO>
  inline std::enable_if_t<(N == 1), void> apply_mask(Arguments& args) {
    // multiply by softmax scale
    sfm_qk_ *= args.sfm_scale;

    if (seq_start_kv_ + kSgSeqKV > args.kv_len) {
      Vec<uint32_t, kSgSeqKV> seq_id(seq_start_kv_, 1);
      __ESIMD_NS::simd_mask<kSgSeqKV> mask = seq_id >= args.kv_len;
      sfm_qk_.merge(kNegInfinity, mask);
    }
  }

  template <int N = kSgSeqQO>
  inline std::enable_if_t<(N > 1), void> apply_mask(Arguments& args) {
    // multiply by softmax scale
    sfm_qk_ *= args.sfm_scale;

// TODO: maybe use cbit to replace this if condition
#pragma unroll
    for (uint32_t i = 0; i < kSgSeqKV; ++i) {
      uint32_t seq_offset_kv = seq_start_kv_ + i;
      if (seq_offset_kv >= args.kv_len) {
        sfm_qk_.template select<kSgSeqQO, kSgSeqKV>(i) = kNegInfinity;
      }
    }
  }

  //
  // Compute softmax
  //
  inline void softmax(sycl::nd_item<2>& ndi, Arguments& args) {
    // compute max_kv
    auto max_kv = row_reduce<BinaryOp::MAX, Accum, kSgSeqQO, kSgSeqKV>(sfm_qk_);
    if constexpr (kNumSgX > 1) {
      // save thread max_kv to SLM
      store_local(slm_offset_store_, max_kv);
      barrier();

      // load all thread max values
      auto tmp = load_local<Accum, kSgSeqQO * kNumSgX>(slm_offset_load_);
      max_kv = col_reduce<BinaryOp::MAX, Accum, kNumSgX, kSgSeqQO>(tmp);
    }
    max_kv = max(max_kv, softmax_m_);

    // handle case when max_kv == -inf
    max_kv.merge(0.0f, max_kv == kNegInfinity);

    // correct old sum
    softmax_s_ *= exp(softmax_m_ - max_kv);

    // compute p = exp(s - m)
#pragma unroll
    for (uint32_t i = 0; i < kSgSeqQO; ++i) {
      auto sfm_qk_sub = sfm_qk_.template select<kSgSeqKV, 1>(i * kSgSeqKV);
      sfm_qk_sub = sfm_qk_sub - max_kv[i];
    }
    sfm_qk_ = exp(sfm_qk_);

    // compute sum_kv
    auto sum_kv = row_reduce<BinaryOp::SUM, Accum, kSgSeqQO, kSgSeqKV>(sfm_qk_);
    if constexpr (kNumSgX > 1) {
      // need sync here before the next writing
      barrier();
      // save thread sum_kv to SLM
      store_local(slm_offset_store_, sum_kv);
      barrier();

      // load all thread sum values
      auto tmp = load_local<Accum, kSgSeqQO * kNumSgX>(slm_offset_load_);
      sum_kv = col_reduce<BinaryOp::SUM, Accum, kNumSgX, kSgSeqQO>(tmp);
    }
    sum_kv += softmax_s_;

    // rescale qk by dividing sum_kv
    // #pragma unroll
    //     for (uint32_t i = 0; i < kSgSeqQO; ++i) {
    //       auto sfm_qk_sub = sfm_qk_.template select<kSgSeqKV, 1>(i *
    //       kSgSeqKV); sfm_qk_sub = sfm_qk_sub / sum_kv[i];
    //     }

    // rescale out
#pragma unroll
    for (uint32_t i = 0; i < kSgSeqQO; ++i) {
      auto accum_o_sub = accum_o_.template select<kHeadSize, 1>(i * kHeadSize);
      float inverse = exp(softmax_m_[i] - max_kv[i]);
      accum_o_sub = accum_o_sub * inverse;
    }

    // update softmax max and sum
    softmax_m_ = max_kv;
    softmax_s_ = sum_kv;
  }

  //
  // Compute qkv = qk * value
  //
  inline void gemm_qkv(sycl::nd_item<2>& ndi, Arguments& args) {
    if constexpr (kUseMask) {
      if (mask_qk_valid_ == 0) {
        return;
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < kSgSeqKV; ++j) {
      uint32_t seq_offset_kv = seq_start_kv_ + j;
      if (seq_offset_kv < args.kv_len) {
        uint32_t offset = offset_kv_ + seq_offset_kv * kSeqStrideKV;
        Vec<T, kHeadSize> tmp_v;
        // load value
#pragma unroll
        for (uint32_t l = 0; l < kHeadSize - kRemainLoadLen; l += kLoadLen) {
          // load value with length of kLoadLen
          tmp_v.template select<kLoadLen, 1>(l) =
              load_global<T, kLoadLen>(args.value, offset + l);
        }

        if constexpr (kRemainLoadLen) {
          tmp_v.template select<kRemainLoadLen, 1>(kHeadSize - kRemainLoadLen) =
              load_global<T, kRemainLoadLen>(
                  args.value, offset + kHeadSize - kRemainLoadLen);
        }

        // compute qk * value
#pragma unroll
        for (uint32_t i = 0; i < kSgSeqQO; ++i) {
          Accum tmp_qk_one = sfm_qk_.template select<1, 1>(i * kSgSeqKV + j)[0];
#pragma unroll
          for (uint32_t s = 0; s < kHeadSize - kRemainHeadSimd; s += kSimd) {
            auto o_sub = accum_o_.template select<kSimd, 1>(i * kHeadSize + s);
            auto v_sub = tmp_v.template select<kSimd, 1>(s);
            o_sub += tmp_qk_one * v_sub;
          }

          if constexpr (kRemainHeadSimd > 0) {
            auto o_tail = accum_o_.template select<kRemainHeadSimd, 1>(
                i * kHeadSize + kHeadSize - kRemainHeadSimd);
            auto v_tail = tmp_v.template select<kRemainHeadSimd, 1>(
                kHeadSize - kRemainHeadSimd);
            o_tail += tmp_qk_one * v_tail;
          }
        }
      }
    }
  }

  // TODO: maybe move seqQO loop
  inline void process_out(sycl::nd_item<2>& ndi, Arguments& args) {
    //  rescale output: accum_o_ /= softmax_s, then save accum_out of each
    //  thread to SLM
    barrier();
#pragma unroll
    for (uint32_t i = 0; i < kSgSeqQO; ++i) {
      uint32_t offset =
          ((sg_idy_ * kSgSeqQO + i) * kNumSgX + sg_idx_) * kHeadSize;
      float inverse = 1 / softmax_s_[i];
      for (uint32_t l = 0; l < kHeadSize - kRemainLoadLen; l += kLoadLen) {
        Vec<Accum, kLoadLen> accum_o_sub =
            (accum_o_.template select<kLoadLen, 1>(i * kHeadSize + l)) *
            inverse;
        store_local(slm_out_ptr + (offset + l) * sizeof(Accum), accum_o_sub);
      }
      if constexpr (kRemainLoadLen) {
        Vec<Accum, kRemainLoadLen> accum_o_sub =
            (accum_o_.template select<kRemainLoadLen, 1>(
                i * kHeadSize + kHeadSize - kRemainLoadLen)) *
            inverse;
        store_local(
            slm_out_ptr + (offset + kHeadSize - kRemainLoadLen) * sizeof(Accum),
            accum_o_sub);
      }
    }
    barrier();

    constexpr uint32_t kSgHeadSize = kHeadSize / kNumSgX;
    constexpr uint32_t kMaxStoreLen =
        (ArchConfig::kMaxStoreBytes / sizeof(T)) > kSgHeadSize
        ? kSgHeadSize
        : (ArchConfig::kMaxStoreBytes / sizeof(T));
    constexpr uint32_t kStoreLen = get_max_power_of_2<kMaxStoreLen>();
    constexpr uint32_t kRemainStoreLen = kSgHeadSize % kStoreLen;

    // temporary storage for out

    // Store kSgSeqQO lines one by one
#pragma unroll
    for (uint32_t i = 0; i < kSgSeqQO; ++i) {
      uint32_t seq_offset_qo = seq_start_qo_ + i;
      if (seq_offset_qo < args.qo_len) {
        Vec<Accum, kSgHeadSize> tmp_o = 0.0f;

        uint32_t offset = (sg_idy_ * kSgSeqQO + i) * kNumSgX * kHeadSize +
            sg_idx_ * kSgHeadSize;
#pragma unroll
        for (uint32_t j = 0; j < kNumSgX; ++j) {
#pragma unroll
          for (uint32_t l = 0; l < kSgHeadSize - kRemainStoreLen;
               l += kStoreLen) {
            auto tmp_o_sub = tmp_o.template select<kStoreLen, 1>(l);
            tmp_o_sub += load_local<Accum, kStoreLen>(
                slm_out_ptr + (offset + j * kHeadSize + l) * sizeof(Accum));
          }

          if constexpr (kRemainStoreLen > 0) {
            auto tmp_o_sub = tmp_o.template select<kRemainStoreLen, 1>(
                kSgHeadSize - kRemainStoreLen);
            tmp_o_sub += load_local<Accum, kRemainStoreLen>(
                slm_out_ptr +
                (offset + j * kHeadSize + kSgHeadSize - kRemainStoreLen) *
                    sizeof(Accum));
          }
        }
        // store out with length of kStoreLen
        offset =
            offset_qo_ + seq_offset_qo * kSeqStrideQO + sg_idx_ * kSgHeadSize;
        Vec<T, kSgHeadSize> tmp_o_t = tmp_o;
#pragma unroll
        for (uint32_t l = 0; l < kSgHeadSize - kRemainStoreLen;
             l += kStoreLen) {
          Vec<T, kStoreLen> tmp_o_sub =
              tmp_o_t.template select<kStoreLen, 1>(l);
          store_global(args.out, offset + l, tmp_o_sub);
        }

        if constexpr (kRemainStoreLen > 0) {
          Vec<T, kRemainStoreLen> tmp_o_sub =
              tmp_o_t.template select<kRemainStoreLen, 1>(
                  kSgHeadSize - kRemainStoreLen);
          store_global(
              args.out, offset + kSgHeadSize - kRemainStoreLen, tmp_o_sub);
        }
      }
    }
  }
};

template <typename Policy, class T>
inline cgf_t launch_fused_mha_impl(
    T* query,
    T* key,
    T* value,
    T* out,
    uint8_t* mask,
    uint32_t num_batches,
    uint32_t qo_len,
    uint32_t kv_len,
    float sfm_scale) {
  using Kernel = FusedMha<Policy, T>;
  auto nd_range = Kernel::nd_range(num_batches, qo_len);
  cgf_t kernel_func = [=](sycl::handler& cgh) {
    cgh.parallel_for<FusedMha<Policy, T>>(
        nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
          typename Kernel::Arguments args{
              query, key, value, out, mask, qo_len, kv_len, sfm_scale};
          Kernel{}(ndi, args);
        });
  };
  return kernel_func;
}

template <bool kHeadFirst, class T>
inline cgf_t launch_phi_fused_mha(
    T* query,
    T* key,
    T* value,
    T* out,
    uint8_t* mask,
    uint32_t num_batches,
    uint32_t qo_len,
    uint32_t kv_len) {
  if (qo_len == 1) {
    using policy = esimd::PhiSingleQueryPolicy<kHeadFirst>;
    float sfm_scale = sycl::rsqrt(float(policy::kHeadSize));
    return launch_fused_mha_impl<policy, T>(
        query, key, value, out, mask, num_batches, qo_len, kv_len, sfm_scale);
  } else {
    using policy = esimd::PhiMultiQueryPolicy<kHeadFirst>;
    float sfm_scale = sycl::rsqrt(float(policy::kHeadSize));
    return launch_fused_mha_impl<policy, T>(
        query, key, value, out, mask, num_batches, qo_len, kv_len, sfm_scale);
  }
}

template <bool kHeadFirst, class T>
inline cgf_t launch_phi3small_fused_mha(
    T* query,
    T* key,
    T* value,
    T* out,
    uint8_t* mask,
    uint32_t num_batches,
    uint32_t qo_len,
    uint32_t kv_len) {
  if (qo_len == 1) {
    using policy = esimd::Phi3SmallSingleQueryPolicy<kHeadFirst>;
    float sfm_scale = 1.0 / float(policy::kHeadSize);
    return launch_fused_mha_impl<policy, T>(
        query, key, value, out, mask, num_batches, qo_len, kv_len, sfm_scale);
  } else {
    using policy = esimd::Phi3SmallMultiQueryPolicy<kHeadFirst>;
    float sfm_scale = 1.0 / float(policy::kHeadSize);
    return launch_fused_mha_impl<policy, T>(
        query, key, value, out, mask, num_batches, qo_len, kv_len, sfm_scale);
  }
}

} // namespace esimd
