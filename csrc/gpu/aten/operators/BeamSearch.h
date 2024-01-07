#pragma once

#include <ATen/record_function.h>
#include "Reduce.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {
// struct for sorted topk seq
template <typename T, int MAX_K>
struct TopK {
  int p[MAX_K];
  T u[MAX_K];

  void insert(T elem, int elem_id) {
    if (elem > u[MAX_K - 1] || (p[MAX_K - 1] == -1) ||
        ((elem == u[MAX_K - 1]) && (elem_id < p[MAX_K - 1]))) {
      // if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id <
      // p[MAX_K-1])))
      u[MAX_K - 1] = elem;
      p[MAX_K - 1] = elem_id;
    }

    for (int k = MAX_K - 2; k >= 0; --k) {
      if ((u[k + 1] > u[k]) || (p[k] == -1) ||
          ((u[k + 1] == u[k]) && (p[k + 1] < p[k]))) {
        // if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
        T u2 = u[k];
        int p2 = p[k];
        u[k] = u[k + 1];
        p[k] = p[k + 1];
        u[k + 1] = u2;
        p[k + 1] = p2;
      }
    }
  }

  void init() const {
    for (int i = 0; i < MAX_K; i++) {
      p[i] = -1;
      u[i] = std::numeric_limits<T>::lowest();
    }
  }
};

template <typename T, int MAX_K>
inline TopK<T, MAX_K> reduce_topk_op(
    const TopK<T, MAX_K>& a,
    const TopK<T, MAX_K>& b) {
  TopK<T, MAX_K> res = a;
  for (int i = 0; i < MAX_K; ++i)
    res.insert(b.u[i], b.p[i]);
  return res;
}

template <typename T, typename item_t, typename ReductionOp>
inline T group_reduce_op(
    item_t item,
    int wg_size,
    dpcpp_local_ptr<void> shared,
    T value,
    ReductionOp reduce) {
  dpcpp_local_ptr<T> shared_(shared);
  int l_x = item.get_local_linear_id();
  int dim_x = wg_size;
  auto sg = item.get_sub_group();
  int sg_size = sg.get_local_range()[0];
  int sg_lid = sg.get_local_linear_id();
  int sg_gid = sg.get_group_linear_id();
  int sg_range = sg.get_group_range()[0];

  for (int offset = 1; offset < sg_size; offset <<= 1) {
    T other = sg.shuffle_down(value, offset);
    value = reduce(value, other);
  }

  if (sg_lid == 0) {
    shared_[sg_gid] = value;
  }
  item.barrier(dpcpp_local_fence);

  if (sg_range <= sg_size) {
    // sub-group reduce
    if (l_x < sg_size) {
      value = shared_[l_x];
    }

    if (sg_gid == 0 && sg_lid < sg_range) {
      value = shared_[sg_lid];
      for (int offset = 1; offset < sg_range; offset <<= 1) {
        T other = sg.shuffle_down(value, offset);
        value = reduce(value, other);
      }
    }
  } else {
    // work item tree reduce
    if (l_x < sg_range) {
      value = shared_[l_x];
    }

    for (int offset = sg_range / 2; offset > 0; offset >>= 1) {
      if (l_x < offset) {
        T other = shared_[l_x + offset];
        value = reduce(value, other);
        shared_[l_x] = value;
      }
      item.barrier(dpcpp_local_fence);
    }
  }
  return value;
}

template <typename scalar_t, int32_t MAX_K>
struct BeamSearchTopkStage1KernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    const int32_t sentence_id = item.get_group(0);
    const int32_t wi_id = item.get_local_id(0);
    const int32_t chunk = (vocab_size + num_wg_per_beam - 1) / num_wg_per_beam;
    const int32_t start = chunk * item.get_group(1);
    int32_t end = start + chunk;
    end = (end > vocab_size) ? vocab_size : end;

    int32_t sentence_offset = sentence_id * vocab_size;

    TopK<scalar_t, MAX_K> partial;

    for (int32_t i = 0; i < MAX_K; i++) {
      partial.p[i] = -1;
      partial.u[i] = std::numeric_limits<scalar_t>::lowest();
    }

    for (int elem_id = start + wi_id; elem_id < end; elem_id += wg_size) {
      scalar_t elem = scores[sentence_offset + elem_id];
      partial.insert(elem, elem_id); // work item reduce
    }

    using arg_vec_t = at::detail::Array<TopK<scalar_t, MAX_K>, 1>;
    arg_vec_t value;
    value[0] = partial;
    auto combine = [=](TopK<scalar_t, MAX_K> value,
                       TopK<scalar_t, MAX_K> other) -> TopK<scalar_t, MAX_K> {
      return reduce_topk_op<scalar_t, MAX_K>(value, other);
    };
    value = xpu::dpcpp::detail::group_reduce<
        TopK<scalar_t, MAX_K>,
        decltype(item),
        decltype(combine),
        1>(item, wg_size, shared, value, combine); // work group reduce
    TopK<scalar_t, MAX_K> total = value[0];

    if (wi_id == 0) {
      int32_t output_offset =
          item.get_group(0) * item.get_group_range(1) * MAX_K +
          item.get_group(1) * MAX_K;
      for (int32_t i = 0; i < MAX_K; i++) {
        tmp_scores[output_offset + i] = total.u[i];
        tmp_idx[output_offset + i] = total.p[i] + sentence_id * vocab_size;
        // global index
      }
    } // write to tmp buffer
  }
  BeamSearchTopkStage1KernelFunctor(
      const scalar_t* scores_,
      scalar_t* tmp_scores_,
      int64_t* tmp_idx_,
      int64_t end_id_,
      int64_t vocab_size_,
      int64_t beam_size_,
      int64_t batch_size_,
      int32_t num_wg_per_beam_,
      int32_t wg_size_,
      dpcpp_local_acc_t<char> shared_)
      : scores(scores_),
        tmp_scores(tmp_scores_),
        tmp_idx(tmp_idx_),
        end_id(end_id_),
        vocab_size(vocab_size_),
        beam_size(beam_size_),
        batch_size(batch_size_),
        num_wg_per_beam(num_wg_per_beam_),
        wg_size(wg_size_),
        shared(shared_) {}

 private:
  const scalar_t* scores;
  scalar_t* tmp_scores;
  int64_t* tmp_idx;
  int64_t end_id;
  int64_t vocab_size;
  int64_t beam_size;
  int64_t batch_size;
  int32_t num_wg_per_beam;
  int32_t wg_size;
  dpcpp_local_acc_t<char> shared;
};

// reduce to 2*beam_size*num_wg_per_beam for each beam
// i.e. 2*beam_size*beam_size*num_wg_per_beam for each batch
template <typename scalar_t, int32_t MAX_K>
void beam_search_topk_stage1(
    const scalar_t* scores,
    scalar_t* tmp_scores,
    int64_t* tmp_idx,
    int64_t end_id,
    int64_t vocab_size,
    int64_t beam_size,
    int64_t batch_size,
    int32_t num_wg_per_beam) {
  int32_t wg_size = dpcppGpuHWThreadsPerEU() * dpcppMaxSubGroupSize();
  int slm_size = (sizeof(int) + sizeof(scalar_t)) * MAX_K * (wg_size);

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<char> shared(slm_size, cgh);
    BeamSearchTopkStage1KernelFunctor<scalar_t, MAX_K> kfn(
        scores,
        tmp_scores,
        tmp_idx,
        end_id,
        vocab_size,
        beam_size,
        batch_size,
        num_wg_per_beam,
        wg_size,
        shared);
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(batch_size * beam_size * wg_size, num_wg_per_beam),
            sycl::range<2>(wg_size, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, int32_t MAX_K>
struct BeamSearchTopkStage2KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int32_t sentence_id = item.get_group(0);
    const int32_t wi_id = item.get_local_id(0);

    TopK<scalar_t, MAX_K> partial;
    for (int i = 0; i < MAX_K; ++i) {
      partial.p[i] = -1;
      partial.u[i] = std::numeric_limits<scalar_t>::lowest();
    }

    if (wi_id < num_wg_per_beam) {
      int32_t offset = sentence_id * MAX_K * num_wg_per_beam + wi_id * MAX_K;
      for (int32_t i = 0; i < 2 * beam_size; i++) {
        partial.u[i] = tmp_scores[offset + i];
        partial.p[i] = tmp_idx[offset + i];
      } // each partial inner is sorted
    }

    using arg_vec_t = at::detail::Array<TopK<scalar_t, MAX_K>, 1>;
    arg_vec_t value;
    value[0] = partial;
    auto combine = [=](TopK<scalar_t, MAX_K> value,
                       TopK<scalar_t, MAX_K> other) -> TopK<scalar_t, MAX_K> {
      return reduce_topk_op<scalar_t, MAX_K>(value, other);
    };
    value = xpu::dpcpp::detail::group_reduce<
        TopK<scalar_t, MAX_K>,
        decltype(item),
        decltype(combine),
        1>(item, num_wg_per_beam, shared, value, combine);
    TopK<scalar_t, MAX_K> total = value[0];

    if (wi_id == 0) {
      int32_t offset = sentence_id * 2 * beam_size;
      for (int32_t i = 0; i < MAX_K; i++) {
        if (i < 2 * beam_size) {
          out_scores[offset + i] = total.u[i];
          out_idx[offset + i] = total.p[i];
        }
      }
    }
  }
  BeamSearchTopkStage2KernelFunctor(
      scalar_t* tmp_scores,
      int64_t* tmp_idx,
      scalar_t* out_scores,
      int64_t* out_idx,
      int64_t vocab_size,
      int64_t beam_size,
      int64_t batch_size,
      int32_t num_wg_per_beam,
      dpcpp_local_acc_t<char> shared)
      : tmp_scores(tmp_scores),
        tmp_idx(tmp_idx),
        out_scores(out_scores),
        out_idx(out_idx),
        vocab_size(vocab_size),
        beam_size(beam_size),
        batch_size(batch_size),
        num_wg_per_beam(num_wg_per_beam),
        shared(shared) {}

 private:
  scalar_t* tmp_scores;
  int64_t* tmp_idx;
  scalar_t* out_scores;
  int64_t* out_idx;
  int64_t vocab_size;
  int64_t beam_size;
  int64_t batch_size;
  int32_t num_wg_per_beam;
  dpcpp_local_acc_t<char> shared;
};

// 2*beam_size*num_wg_per_beam --> 2*beam_size for each beam
template <typename scalar_t, int32_t MAX_K>
void beam_search_topk_stage2(
    scalar_t* tmp_scores,
    int64_t* tmp_idx,
    scalar_t* out_scores,
    int64_t* out_idx,
    int64_t vocab_size,
    int64_t beam_size,
    int64_t batch_size,
    int32_t num_wg_per_beam) {
  int slm_size = (sizeof(int) + sizeof(scalar_t)) * MAX_K * num_wg_per_beam;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<char> shared(slm_size, cgh);
    BeamSearchTopkStage2KernelFunctor<scalar_t, MAX_K> kfn(
        tmp_scores,
        tmp_idx,
        out_scores,
        out_idx,
        vocab_size,
        beam_size,
        batch_size,
        num_wg_per_beam,
        shared);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(batch_size * beam_size * num_wg_per_beam),
            sycl::range<1>(num_wg_per_beam)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T>
inline T apply_length_penalty(
    T log_prob,
    int64_t length,
    float length_penalty) {
  // score = log(prob) / (length)^length_penalty.
  if (length_penalty == 0.0f || length == 1) {
    return log_prob;
  }
  return log_prob / static_cast<T>(Numerics<T>::pow(length, length_penalty));
}

template <typename T>
struct TopK_2 {
  int p = -1;
  T u = std::numeric_limits<T>::lowest();

  void insert(T elem, int elem_id) {
    if (elem > u) {
      u = elem;
      p = elem_id;
    }
  }
};

template <typename T>
inline TopK_2<T> reduce_topk_op_2(const TopK_2<T>& a, const TopK_2<T>& b) {
  return a.u > b.u ? a : b;
}

template <typename scalar_t, int32_t MAX_K>
struct BatchTopkKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int32_t wi_id = item.get_local_id(0);
    const int32_t wg_id = item.get_group(0); // each work group handle a batch
    int32_t wg_offset = wg_id * length_per_wg;

    // init
    if (wi_id < beam_size) {
      top_score[wg_id * beam_size + wi_id] = 0;
      top_token[wg_id * beam_size + wi_id] = pad_token_id;
      top_beams[wg_id * beam_size + wi_id] = 0;
    }

    if (finished[wg_id]) {
      return;
    }

    if (beam_hyps_num_beams[wg_id] == 0 && wi_id == 0) {
      beam_hyps_min_normed_scores[wg_id] = std::numeric_limits<scalar_t>::max();
    }

    TopK<scalar_t, MAX_K> partial;
    for (int i = 0; i < MAX_K; ++i) {
      partial.p[i] = -1;
      partial.u[i] = std::numeric_limits<scalar_t>::lowest();
    }

    for (int elem_id = wi_id; elem_id < length_per_wg; elem_id += wg_size) {
      int32_t i = elem_id % beam_size;
      scalar_t elem = topk_tmp_val_buf[wg_offset + elem_id];

      int elem_idx = elem_id; // x[elem_id];
      partial.insert(elem, elem_idx);
    }

    // beam_size*beam_size*2 reduce to top 2*beam_size
    using arg_vec_t = at::detail::Array<TopK<scalar_t, MAX_K>, 1>;
    arg_vec_t value;
    value[0] = partial;
    auto combine = [=](TopK<scalar_t, MAX_K> value,
                       TopK<scalar_t, MAX_K> other) -> TopK<scalar_t, MAX_K> {
      return reduce_topk_op<scalar_t, MAX_K>(value, other);
    };
    value = xpu::dpcpp::detail::group_reduce<
        TopK<scalar_t, MAX_K>,
        decltype(item),
        decltype(combine),
        1>(item, wg_size, shared, value, combine);
    TopK<scalar_t, MAX_K> total = value[0];

    if (wi_id == 0) {
      int32_t wg_offset_k = wg_id * beam_size;
      int32_t selected_beams = 0; // the counter of beams selected without EOS

      for (int i = 0; i < MAX_K; ++i) {
        if (topk_tmp_id_buf[wg_offset + total.p[i]] % vocab_size ==
            eos_token_id) {
          if (i >= beam_size) {
            // do nothing
          } else {
            const int32_t global_batch_idx = wg_id;
            const scalar_t normed_score = apply_length_penalty(
                total.u[i], cur_len + max_in_seq_len, length_penalty);
            const int num_beam = beam_hyps_num_beams[global_batch_idx];
            int beam_idx = num_beam;
            // the candidate pool is full
            if (num_beam == beam_size) {
              if (normed_score <
                  beam_hyps_min_normed_scores[global_batch_idx]) {
                // if the current score is less than the min score in
                // candidate list, exit
                selected_beams = beam_size;
                break;
              } else {
                // replace the min score in candidate with new score
                for (int j = 0; j < beam_size; j++) {
                  // find the min score, replace with current score, and sort
                  // the new min score
                  if (beam_hyps_normed_scores
                          [global_batch_idx * (beam_size * 2) + j] ==
                      beam_hyps_min_normed_scores[global_batch_idx]) {
                    beam_idx = j; // find current min score index
                    beam_hyps_num_beams[global_batch_idx]--;

                    beam_hyps_min_normed_scores[global_batch_idx] =
                        std::numeric_limits<scalar_t>::max();
                    beam_hyps_normed_scores
                        [global_batch_idx * (beam_size * 2) + j] = normed_score;
                    // replace with current score
                    for (int l = 0; l < beam_size; l++) {
                      beam_hyps_min_normed_scores[global_batch_idx] =
                          Numerics<scalar_t>::min(
                              beam_hyps_min_normed_scores[global_batch_idx],
                              beam_hyps_normed_scores
                                  [global_batch_idx * (beam_size * 2) + l]);
                    } // sort for the min score
                    break;
                  }
                }
              }
            }
            // when the list is full, update the one to be replaced
            // not full, add new sentence to the end
            const int tgt_id_offset =
                (wg_id * (beam_size * 2) + beam_idx) * max_out_seq_len;
            beam_hyps_output_ids_tgt[tgt_id_offset + cur_len] =
                eos_token_id; // update output ids for last time step

            int prev_id =
                (topk_tmp_id_buf[wg_offset + total.p[i]] / vocab_size) %
                beam_size; // which beam
            for (int j = cur_len - 1; j >= 0; j--) {
              // generate the new added sentence
              const int src_idx =
                  j * batch_size * beam_size + wg_id * beam_size + prev_id;

              beam_hyps_output_ids_tgt[tgt_id_offset + j] =
                  beam_hyps_output_ids_src[src_idx]; // copy (time_step-1) to
                                                     // output ids
              prev_id = beam_hyps_beam_ids_src[src_idx]; // query last time
                                                         // step beam id
            }
            const int tgt_beam_idx =
                global_batch_idx * (beam_size * 2) + beam_idx;
            beam_hyps_sequence_lengths_tgt[tgt_beam_idx] =
                cur_len + 1; // +1 for eos token
            beam_hyps_normed_scores[tgt_beam_idx] = normed_score;
            beam_hyps_min_normed_scores[global_batch_idx] =
                Numerics<scalar_t>::min(
                    beam_hyps_min_normed_scores[global_batch_idx],
                    beam_hyps_normed_scores[tgt_beam_idx]);

            beam_hyps_num_beams[global_batch_idx]++;
            beam_hyps_score[tgt_beam_idx] =
                topk_tmp_val_buf[wg_offset + total.p[i]];
          }
        } else if (i < 2 * beam_size) {
          int64_t global_index = topk_tmp_id_buf[wg_offset + total.p[i]];
          top_token[wg_offset_k + selected_beams] = global_index % vocab_size;
          top_beams[wg_offset_k + selected_beams] =
              global_index / vocab_size % beam_size;
          top_score[wg_offset_k + selected_beams] =
              topk_tmp_val_buf[wg_offset + total.p[i]];
          selected_beams++;
        }
        if (selected_beams >= beam_size) {
          break;
        }
      }
      if (beam_hyps_num_beams[wg_id] == beam_size) {
        if (early_stopping) {
          finished[wg_id] = true;
        } else {
          scalar_t highest_attainable_score = apply_length_penalty(
              top_score[wg_offset_k], cur_len + max_in_seq_len, length_penalty);
          if (beam_hyps_min_normed_scores[wg_id] >= highest_attainable_score) {
            finished[wg_id] = true;
          }
        }
      }
    }
  }
  BatchTopkKernelFunctor(
      const scalar_t* topk_tmp_val_buf_,
      const int64_t* topk_tmp_id_buf_,
      scalar_t* top_score_,
      int64_t* top_token_,
      int64_t* top_beams_,
      int64_t pad_token_id_,
      int64_t eos_token_id_,
      bool* finished_,
      int64_t* beam_hyps_num_beams_,
      scalar_t* beam_hyps_min_normed_scores_,
      scalar_t* beam_hyps_normed_scores_,
      int64_t* beam_hyps_output_ids_tgt_,
      int64_t* beam_hyps_output_ids_src_,
      int64_t* beam_hyps_beam_ids_src_,
      int64_t* beam_hyps_sequence_lengths_tgt_,
      scalar_t* beam_hyps_score_,
      const float length_penalty_,
      const int length_per_wg_,
      const int64_t beam_size_,
      const int64_t batch_size_,
      const int64_t vocab_size_,
      int64_t cur_len_,
      int64_t max_in_seq_len_,
      int64_t max_out_seq_len_,
      bool early_stopping_,
      int32_t wg_size_,
      dpcpp_local_acc_t<unsigned char> shared_)
      : topk_tmp_val_buf(topk_tmp_val_buf_),
        topk_tmp_id_buf(topk_tmp_id_buf_),
        top_score(top_score_),
        top_token(top_token_),
        top_beams(top_beams_),
        pad_token_id(pad_token_id_),
        eos_token_id(eos_token_id_),
        finished(finished_),
        beam_hyps_num_beams(beam_hyps_num_beams_),
        beam_hyps_min_normed_scores(beam_hyps_min_normed_scores_),
        beam_hyps_normed_scores(beam_hyps_normed_scores_),
        beam_hyps_output_ids_tgt(beam_hyps_output_ids_tgt_),
        beam_hyps_output_ids_src(beam_hyps_output_ids_src_),
        beam_hyps_beam_ids_src(beam_hyps_beam_ids_src_),
        beam_hyps_sequence_lengths_tgt(beam_hyps_sequence_lengths_tgt_),
        beam_hyps_score(beam_hyps_score_),
        length_penalty(length_penalty_),
        length_per_wg(length_per_wg_),
        beam_size(beam_size_),
        batch_size(batch_size_),
        vocab_size(vocab_size_),
        cur_len(cur_len_),
        max_in_seq_len(max_in_seq_len_),
        max_out_seq_len(max_out_seq_len_),
        early_stopping(early_stopping_),
        wg_size(wg_size_),
        shared(shared_) {}

 private:
  const scalar_t* topk_tmp_val_buf;
  const int64_t* topk_tmp_id_buf;
  scalar_t* top_score;
  int64_t* top_token;
  int64_t* top_beams;
  int64_t pad_token_id;
  int64_t eos_token_id;
  bool* finished;
  int64_t* beam_hyps_num_beams;
  scalar_t* beam_hyps_min_normed_scores;
  scalar_t* beam_hyps_normed_scores;
  int64_t* beam_hyps_output_ids_tgt;
  int64_t* beam_hyps_output_ids_src;
  int64_t* beam_hyps_beam_ids_src;
  int64_t* beam_hyps_sequence_lengths_tgt;
  scalar_t* beam_hyps_score;
  const float length_penalty;
  const int length_per_wg;
  const int64_t beam_size;
  const int64_t batch_size;
  const int64_t vocab_size;
  int64_t cur_len;
  int64_t max_in_seq_len;
  int64_t max_out_seq_len;
  bool early_stopping;
  int32_t wg_size;
  dpcpp_local_acc_t<unsigned char> shared;
};

// 1, 2*beam_size*beam_size --> 2*beam_size for each batch
// 2, 2*beam_size --> beam_size
template <typename scalar_t, int32_t MAX_K>
void batch_topk_kernel(
    const scalar_t* topk_tmp_val_buf,
    const int64_t*
        topk_tmp_id_buf, // len = batch_size x beam_size x beam_size x 2
    scalar_t* top_score, // out
    int64_t* top_token, // out for for topk (global id)
    int64_t* top_beams, // out
    int64_t pad_token_id,
    int64_t eos_token_id,
    bool* finished, // [batch_size]
    // parameters for beam_hyps
    int64_t* beam_hyps_num_beams, // [batch_size] number for eos candidates
    scalar_t* beam_hyps_min_normed_scores, // [batch_size]
    scalar_t* beam_hyps_normed_scores, // [batch_size * 2 * beam_size], store
                                       // the norm scores for candidates
    int64_t* beam_hyps_output_ids_tgt, // [batch_size * 2 * beam_size, max_seq],
                                       // cadidate output sentence
    int64_t* beam_hyps_output_ids_src, // [max_seq, batch_size * beam_size],
                                       // the out_buffer
    int64_t* beam_hyps_beam_ids_src, // [max_seq, batch_size * beam_size]
    int64_t* beam_hyps_sequence_lengths_tgt, // [batch_size * 2 * beam_size]
    scalar_t* beam_hyps_score, // [batch_size * 2 * beam_size]
    const float length_penalty,
    const int length_per_wg, // beam_size x beam_size x 2
    const int64_t beam_size,
    const int64_t batch_size,
    const int64_t vocab_size,
    int64_t cur_len,
    int64_t max_in_seq_len,
    int64_t max_out_seq_len,
    bool early_stopping) {
  // one wg handle one batch
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int32_t wg_size = 32;
  int slm_size = (sizeof(int) + sizeof(scalar_t)) * MAX_K * wg_size;
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto shared = dpcpp_local_acc_t<unsigned char>(slm_size, cgh);
    BatchTopkKernelFunctor<scalar_t, MAX_K> kfn(
        topk_tmp_val_buf,
        topk_tmp_id_buf,
        top_score,
        top_token,
        top_beams,
        pad_token_id,
        eos_token_id,
        finished,
        beam_hyps_num_beams,
        beam_hyps_min_normed_scores,
        beam_hyps_normed_scores,
        beam_hyps_output_ids_tgt,
        beam_hyps_output_ids_src,
        beam_hyps_beam_ids_src,
        beam_hyps_sequence_lengths_tgt,
        beam_hyps_score,
        length_penalty,
        length_per_wg,
        beam_size,
        batch_size,
        vocab_size,
        cur_len,
        max_in_seq_len,
        max_out_seq_len,
        early_stopping,
        wg_size,
        shared);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(batch_size * 32), sycl::range<1>(32)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

struct UpdateTokenKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    for (int32_t index = item.get_local_id(0); index < batch_size * beam_size;
         index += wg_size) {
      int32_t offset = time_step * batch_size * beam_size;
      int batch_idx = index / beam_size;

      beam_ids[offset + index] = top_beam_id[index];
      word_ids[offset + index] = top_word_id[index];
    }
  }
  UpdateTokenKernelFunctor(
      int64_t* top_beam_id_,
      int64_t* top_word_id_,
      int64_t* beam_ids_,
      int64_t* word_ids_,
      const int64_t time_step_,
      const int64_t batch_size_,
      const int64_t beam_size_,
      int32_t wg_size_)
      : top_beam_id(top_beam_id_),
        top_word_id(top_word_id_),
        beam_ids(beam_ids_),
        word_ids(word_ids_),
        time_step(time_step_),
        batch_size(batch_size_),
        beam_size(beam_size_),
        wg_size(wg_size_) {}

 private:
  int64_t* top_beam_id;
  int64_t* top_word_id;
  int64_t* beam_ids;
  int64_t* word_ids;
  const int64_t time_step;
  const int64_t batch_size;
  const int64_t beam_size;
  int32_t wg_size;
};

void update_token(
    int64_t* top_beam_id, // in [batch_size * beam_size]
    int64_t* top_word_id, // in
    int64_t* beam_ids, // out [max_out_seq_len, batch_size * beam_size]
    int64_t* word_ids, // out [max_out_seq_len, batch_size * beam_size]
    bool* finished,
    const int64_t time_step,
    const int64_t batch_size,
    const int64_t beam_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int32_t wg_size = 256;

  auto cgf = DPCPP_Q_CGF(cgh) {
    UpdateTokenKernelFunctor kfn(
        top_beam_id,
        top_word_id,
        beam_ids,
        word_ids,
        time_step,
        batch_size,
        beam_size,
        wg_size);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
struct UpdateBeamIndiceKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    int32_t time_step =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    int32_t sentence_id =
        item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

    int32_t beam_id = sentence_id % beam_size;
    int32_t batch_id = sentence_id / beam_size;
    int32_t offset = num_step * batch_size * beam_size;

    if (sentence_id < batch_size * beam_size && time_step < num_step) {
      const scalar_t src_beam = beam_ids[sentence_id];
      const int32_t src_offset =
          batch_size * beam_size * time_step + batch_id * beam_size + src_beam;
      const int32_t out_offset =
          batch_size * beam_size * time_step + batch_id * beam_size + beam_id;

      out_cache_indices[out_offset] =
          (time_step == step) ? beam_id : src_cache_indices[src_offset];
    }
  }
  UpdateBeamIndiceKernelFunctor(
      scalar_t* src_cache_indices_,
      scalar_t* out_cache_indices_,
      int64_t* beam_ids_,
      int32_t step_,
      int32_t beam_size_,
      int32_t batch_size_,
      int32_t num_step_)
      : src_cache_indices(src_cache_indices_),
        out_cache_indices(out_cache_indices_),
        beam_ids(beam_ids_),
        step(step_),
        beam_size(beam_size_),
        batch_size(batch_size_),
        num_step(num_step_) {}

 private:
  scalar_t* src_cache_indices;
  scalar_t* out_cache_indices;
  int64_t* beam_ids;
  int32_t step;
  int32_t beam_size;
  int32_t batch_size;
  int32_t num_step;
};

template <typename scalar_t>
void update_beam_indices_kernel(
    scalar_t* src_cache_indices,
    scalar_t* out_cache_indices,
    int64_t* beam_ids,
    int32_t step,
    int32_t beam_size,
    int32_t batch_size) {
  int32_t num_step = step + 1;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int32_t wg_size = 32;
  int32_t wg_number = (num_step + wg_size - 1) / wg_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    UpdateBeamIndiceKernelFunctor<scalar_t> kfn(
        src_cache_indices,
        out_cache_indices,
        beam_ids,
        step,
        beam_size,
        batch_size,
        num_step);
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(wg_number * wg_size, batch_size * beam_size),
            sycl::range<2>(wg_size, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
struct UpdateNativeBeamIndiceKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    int32_t time_step =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    int32_t sentence_id =
        item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

    int32_t beam_id = sentence_id % beam_size;
    int32_t batch_id = sentence_id / beam_size;
    int32_t offset = num_step * batch_size * beam_size;

    if (sentence_id < batch_size * beam_size && time_step < num_step) {
      const scalar_t src_beam =
          (scalar_t)beam_ids[sentence_id] - batch_id * beam_size;
      // for reference beam search
      const int32_t src_offset =
          batch_size * beam_size * time_step + batch_id * beam_size + src_beam;
      const int32_t out_offset =
          batch_size * beam_size * time_step + batch_id * beam_size + beam_id;

      out_cache_indices[out_offset] =
          (time_step == step) ? beam_id : src_cache_indices[src_offset];
    }
  }
  UpdateNativeBeamIndiceKernelFunctor(
      scalar_t* src_cache_indices_,
      scalar_t* out_cache_indices_,
      int64_t* beam_ids_,
      int32_t step_,
      int32_t beam_size_,
      int32_t batch_size_,
      int32_t num_step_)
      : src_cache_indices(src_cache_indices_),
        out_cache_indices(out_cache_indices_),
        beam_ids(beam_ids_),
        step(step_),
        beam_size(beam_size_),
        batch_size(batch_size_),
        num_step(num_step_) {}

 private:
  scalar_t* src_cache_indices;
  scalar_t* out_cache_indices;
  int64_t* beam_ids;
  int32_t step;
  int32_t beam_size;
  int32_t batch_size;
  int32_t num_step;
};

template <typename scalar_t>
void update_native_beam_indices_kernel(
    scalar_t* src_cache_indices,
    scalar_t* out_cache_indices,
    int64_t* beam_ids,
    int32_t step,
    int32_t beam_size,
    int32_t batch_size) {
  int32_t num_step = step + 1;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int32_t wg_size = 32;
  int32_t wg_number = (num_step + wg_size - 1) / wg_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    UpdateNativeBeamIndiceKernelFunctor<scalar_t> kfn(
        src_cache_indices,
        out_cache_indices,
        beam_ids,
        step,
        beam_size,
        batch_size,
        num_step);

    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(wg_number * wg_size, batch_size * beam_size),
            sycl::range<2>(wg_size, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
struct InsertToCandidateListKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int wg_id = item.get_group(0);
    const int tgt_start_idx =
        beam_hyps_num_beams[wg_id]; // start for seq to add
    // finished batch pass
    if (finished[wg_id]) {
      return;
    }
    for (int i = 0; i < beam_size; i++) {
      if (item.get_local_id(0) == 0) {
        const int src_beam_idx = wg_id * beam_size + i;
        const int tgt_beam_idx = wg_id * beam_size * 2 + i + tgt_start_idx;

        const int length = cur_len - 1;
        // output_ids_src is output_ids
        beam_hyps_output_ids_tgt[(tgt_beam_idx)*max_out_seq_len + length] =
            beam_hyps_output_ids_src
                [length * batch_size * beam_size + src_beam_idx];
        int prev_id = beam_hyps_beam_ids_src
            [length * batch_size * beam_size + src_beam_idx];
        for (int j = length - 1; j >= 0; j--) {
          // copy the complete sentence
          // output_ids_tgt  [bs, beam_size, max_seq_len + 1]
          beam_hyps_output_ids_tgt[(tgt_beam_idx)*max_out_seq_len + j] =
              beam_hyps_output_ids_src
                  [j * batch_size * beam_size + wg_id * beam_size + prev_id];

          prev_id = beam_hyps_beam_ids_src
              [j * batch_size * beam_size + wg_id * beam_size + prev_id];
        }
        beam_hyps_sequence_lengths_tgt[tgt_beam_idx] =
            cur_len; // update target length

        beam_hyps_normed_scores[tgt_beam_idx] = apply_length_penalty(
            top_score[src_beam_idx], max_in_seq_len + length, length_penalty);
        beam_hyps_score[tgt_beam_idx] = top_score[src_beam_idx];
        // update score and norm score

        beam_hyps_num_beams[wg_id]++;
      }
    }
  }
  InsertToCandidateListKernelFunctor(
      int64_t* beam_hyps_num_beams_,
      int64_t* beam_hyps_sequence_lengths_tgt_,
      int64_t* beam_hyps_output_ids_src_,
      int64_t* beam_hyps_output_ids_tgt_,
      int64_t* beam_hyps_beam_ids_src_,
      scalar_t* beam_hyps_score_,
      scalar_t* beam_hyps_normed_scores_,
      scalar_t* top_score_,
      const bool* finished_,
      const float length_penalty_,
      const int64_t max_in_seq_len_,
      const int64_t max_out_seq_len_,
      const int64_t batch_size_,
      const int64_t beam_size_,
      const int64_t cur_len_,
      int32_t wg_size_)
      : beam_hyps_num_beams(beam_hyps_num_beams_),
        beam_hyps_sequence_lengths_tgt(beam_hyps_sequence_lengths_tgt_),
        beam_hyps_output_ids_src(beam_hyps_output_ids_src_),
        beam_hyps_output_ids_tgt(beam_hyps_output_ids_tgt_),
        beam_hyps_beam_ids_src(beam_hyps_beam_ids_src_),
        beam_hyps_score(beam_hyps_score_),
        beam_hyps_normed_scores(beam_hyps_normed_scores_),
        top_score(top_score_),
        finished(finished_),
        length_penalty(length_penalty_),
        max_in_seq_len(max_in_seq_len_),
        max_out_seq_len(max_out_seq_len_),
        batch_size(batch_size_),
        beam_size(beam_size_),
        cur_len(cur_len_),
        wg_size(wg_size_) {}

 private:
  int64_t* beam_hyps_num_beams;
  int64_t* beam_hyps_sequence_lengths_tgt;
  int64_t* beam_hyps_output_ids_src;
  int64_t* beam_hyps_output_ids_tgt;
  int64_t* beam_hyps_beam_ids_src;
  scalar_t* beam_hyps_score;
  scalar_t* beam_hyps_normed_scores;
  scalar_t* top_score;
  const bool* finished;
  const float length_penalty;
  const int64_t max_in_seq_len;
  const int64_t max_out_seq_len;
  const int64_t batch_size;
  const int64_t beam_size;
  const int64_t cur_len;
  int32_t wg_size;
};

template <typename scalar_t>
void insert_to_candidate_list(
    int64_t* beam_hyps_num_beams,
    int64_t* beam_hyps_sequence_lengths_tgt,
    int64_t* beam_hyps_output_ids_src, // [max_seq_len, batch_size, beam_size]
    int64_t* beam_hyps_output_ids_tgt,
    int64_t* beam_hyps_beam_ids_src, // [max_seq_len, batch_size, beam_size]
    scalar_t* beam_hyps_score,
    scalar_t* beam_hyps_normed_scores,
    scalar_t* top_score,
    const bool* finished,
    const float length_penalty,
    const int64_t max_in_seq_len,
    const int64_t max_out_seq_len,
    const int64_t batch_size,
    const int64_t beam_size,
    const int64_t cur_len) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int32_t wg_size = 256;

  auto cgf = DPCPP_Q_CGF(cgh) {
    InsertToCandidateListKernelFunctor<scalar_t> kfn(
        beam_hyps_num_beams,
        beam_hyps_sequence_lengths_tgt,
        beam_hyps_output_ids_src,
        beam_hyps_output_ids_tgt,
        beam_hyps_beam_ids_src,
        beam_hyps_score,
        beam_hyps_normed_scores,
        top_score,
        finished,
        length_penalty,
        max_in_seq_len,
        max_out_seq_len,
        batch_size,
        beam_size,
        cur_len,
        wg_size);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(batch_size * 256), sycl::range<1>(256)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
struct FinalizeKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int32_t wi_id = item.get_local_id(0);
    const int32_t wg_id = item.get_group(0);
    if (wi_id < beam_hyps_num_beams[wg_id]) {
      s_score[wi_id] = beam_hyps_normed_scores[wg_id * wg_size + wi_id];
    } else {
      s_score[wi_id] = std::numeric_limits<scalar_t>::lowest();
    }

    item.barrier(dpcpp_local_fence);

    auto sg = item.get_sub_group();
    int sg_size = sg.get_local_range()[0];
    auto combine = [=](scalar_t value, scalar_t other) -> scalar_t {
      return Numerics<scalar_t>::max(value, other);
    };
    for (int i = 0; i < out_sentence_num; i++) {
      scalar_t value = s_score[wi_id];
      for (int offset = 1; offset < sg_size; offset <<= 1) {
        scalar_t other = sg.shuffle_down(value, offset);
        value = combine(value, other);
      }

      if (wi_id == 0) {
        for (int j = 0; j < beam_size * 2; j++) {
          if (s_score[j] == value) {
            rank[i] = j;
            s_score[j] = std::numeric_limits<scalar_t>::lowest();
            break;
          }
        }
      }
      item.barrier(dpcpp_local_fence);
    }

    if (wi_id < out_sentence_num) {
      sequence_lengths[wg_id * out_sentence_num + wi_id] =
          beam_hyps_sequence_lengths_tgt[wg_id * 2 * beam_size + rank[wi_id]];
    }

    auto total_len = max_in_seq_len + max_out_seq_len;
    for (int32_t index = 0; index < out_sentence_num; index++) {
      // all of the work items will decode one sentence
      for (int32_t t_id = wi_id; t_id < max_out_seq_len; t_id += wg_size) {
        if (t_id < sequence_lengths[wg_id * out_sentence_num + index]) {
          output_ids
              [wg_id * out_sentence_num * total_len + index * total_len +
               max_in_seq_len + t_id] = beam_hyps_output_ids_tgt
                  [wg_id * (beam_size * 2) * max_out_seq_len +
                   rank[index] * max_out_seq_len + t_id];
        } else {
          output_ids
              [wg_id * out_sentence_num * total_len + index * total_len +
               max_in_seq_len + t_id] = pad_token_id;
        }
      }
    }
  }
  FinalizeKernelFunctor(
      int64_t* output_ids_,
      int64_t* sequence_lengths_,
      const int64_t* beam_hyps_output_ids_tgt_,
      const int64_t* beam_hyps_sequence_lengths_tgt_,
      const scalar_t* beam_hyps_normed_scores_,
      const int64_t* beam_hyps_num_beams_,
      const int64_t beam_size_,
      const int64_t batch_size_,
      const int64_t max_in_seq_len_,
      const int64_t max_out_seq_len_,
      const int64_t out_sentence_num_,
      const int64_t pad_token_id_,
      int32_t wg_size_,
      dpcpp_local_acc_t<scalar_t> s_score_,
      dpcpp_local_acc_t<int32_t> rank_)
      : output_ids(output_ids_),
        sequence_lengths(sequence_lengths_),
        beam_hyps_output_ids_tgt(beam_hyps_output_ids_tgt_),
        beam_hyps_sequence_lengths_tgt(beam_hyps_sequence_lengths_tgt_),
        beam_hyps_normed_scores(beam_hyps_normed_scores_),
        beam_hyps_num_beams(beam_hyps_num_beams_),
        beam_size(beam_size_),
        batch_size(batch_size_),
        max_in_seq_len(max_in_seq_len_),
        max_out_seq_len(max_out_seq_len_),
        out_sentence_num(out_sentence_num_),
        pad_token_id(pad_token_id_),
        wg_size(wg_size_),
        s_score(s_score_),
        rank(rank_) {}

 private:
  int64_t* output_ids;
  int64_t* sequence_lengths;
  const int64_t* beam_hyps_output_ids_tgt;
  const int64_t* beam_hyps_sequence_lengths_tgt;
  const scalar_t* beam_hyps_normed_scores;
  const int64_t* beam_hyps_num_beams;
  const int64_t beam_size;
  const int64_t batch_size;
  const int64_t max_in_seq_len;
  const int64_t max_out_seq_len;
  const int64_t out_sentence_num;
  const int64_t pad_token_id;
  int32_t wg_size;
  dpcpp_local_acc_t<scalar_t> s_score;
  dpcpp_local_acc_t<int32_t> rank;
};

// sentences from cadidate list, have eos token at the end. others don't
template <typename scalar_t>
void finalize(
    int64_t* output_ids, // [bs, out_sentence_num, max_seq_len], new tensor for
                         // output
    int64_t* sequence_lengths, // [batch_size, out_sentence_num]
    const int64_t* beam_hyps_output_ids_tgt,
    const int64_t* beam_hyps_sequence_lengths_tgt,
    const scalar_t* beam_hyps_normed_scores, // [bs, 2 * beam_size]
    const int64_t* beam_hyps_num_beams,
    const int64_t beam_size,
    const int64_t batch_size,
    const int64_t max_in_seq_len,
    const int64_t max_out_seq_len,
    const int64_t out_sentence_num,
    const int64_t pad_token_id) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int32_t wg_size = 2 * beam_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto s_score = dpcpp_local_acc_t<scalar_t>(wg_size, cgh);
    auto rank = dpcpp_local_acc_t<int32_t>(out_sentence_num, cgh);
    FinalizeKernelFunctor<scalar_t> kfn(
        output_ids,
        sequence_lengths,
        beam_hyps_output_ids_tgt,
        beam_hyps_sequence_lengths_tgt,
        beam_hyps_normed_scores,
        beam_hyps_num_beams,
        beam_size,
        batch_size,
        max_in_seq_len,
        max_out_seq_len,
        out_sentence_num,
        pad_token_id,
        wg_size,
        s_score,
        rank);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(batch_size * wg_size), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
struct CopyInputToOutputKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int32_t wi_id = item.get_local_id(0);
    int32_t wg_id = item.get_group(0);

    for (int32_t index = wi_id; index < input_len; index += wg_size) {
      output_ids[wg_id * output_len + index] =
          input_ids[(wg_id / out_beams) * beam_size * input_len + index];
    }
  }
  CopyInputToOutputKernelFunctor(
      scalar_t* input_ids,
      scalar_t* output_ids,
      int32_t input_len,
      int32_t output_len,
      int32_t seq_num,
      int32_t beam_size,
      int32_t out_beams,
      int32_t wg_size)
      : input_ids(input_ids),
        output_ids(output_ids),
        input_len(input_len),
        output_len(output_len),
        seq_num(seq_num),
        beam_size(beam_size),
        out_beams(out_beams),
        wg_size(wg_size) {}

 private:
  scalar_t* input_ids;
  scalar_t* output_ids;
  int32_t input_len;
  int32_t output_len;
  int32_t seq_num;
  int32_t beam_size;
  int32_t out_beams;
  int32_t wg_size;
};

template <typename scalar_t>
void copy_input_to_output(
    scalar_t* input_ids,
    scalar_t* output_ids,
    int32_t input_len,
    int32_t output_len,
    int32_t seq_num,
    int32_t beam_size,
    int32_t out_beams) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int32_t wg_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    CopyInputToOutputKernelFunctor<scalar_t> kfn(
        input_ids,
        output_ids,
        input_len,
        output_len,
        seq_num,
        beam_size,
        out_beams,
        wg_size);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(wg_size * seq_num), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at
