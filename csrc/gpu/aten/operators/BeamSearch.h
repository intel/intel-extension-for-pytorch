#pragma once

#include <ATen/record_function.h>
#include "Reduce.h"
#include "comm/Numerics.h"

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
  int32_t wg_size = 128;
  int slm_size = (sizeof(int) + sizeof(scalar_t)) * MAX_K * (wg_size);

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<char> shared(slm_size, cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      const int32_t sentence_id = item.get_group(0);
      const int32_t wi_id = item.get_local_id(0);
      const int32_t chunk =
          (vocab_size + num_wg_per_beam - 1) / num_wg_per_beam;
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
    };

    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(batch_size * beam_size * wg_size, num_wg_per_beam),
            sycl::range<2>(wg_size, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

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
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
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
    };

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
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      const int32_t wi_id = item.get_local_id(0);
      const int32_t wg_id = item.get_group(0); // each work group handle a batch
      int32_t wg_offset = wg_id * length_per_wg;

      // init
      if (wi_id < beam_size) {
        top_score[wg_id * beam_size + beam_size] = 0;
        top_token[wg_id * beam_size + beam_size] = pad_token_id;
        top_beams[wg_id * beam_size + beam_size] = 0;
      }

      if (finished[wg_id]) {
        return;
      }

      if (beam_hyps_num_beams[wg_id] == 0 && wi_id == 0) {
        beam_hyps_min_normed_scores[wg_id] =
            std::numeric_limits<scalar_t>::max();
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
              const float normed_score = apply_length_penalty(
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
                          [global_batch_idx * (beam_size * 2) + j] =
                              normed_score;
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
                  (float)topk_tmp_val_buf[wg_offset + total.p[i]];
            }
          } else if (i < 2 * beam_size) {
            int64_t global_index = topk_tmp_id_buf[wg_offset + total.p[i]];
            top_token[wg_offset_k + selected_beams] = global_index % vocab_size;
            top_beams[wg_offset_k + selected_beams] =
                global_index / vocab_size % beam_size;
            top_score[wg_offset_k + selected_beams] =
                (float)topk_tmp_val_buf[wg_offset + total.p[i]];
            selected_beams++;
          }
          item.barrier(dpcpp_local_fence);
          if (selected_beams >= beam_size) {
            break;
          }
        }
        if (beam_hyps_num_beams[wg_id] == beam_size) {
          if (early_stopping) {
            finished[wg_id] = true;
          } else {
            scalar_t highest_attainable_score = apply_length_penalty(
                top_score[wg_offset_k],
                cur_len + max_in_seq_len,
                length_penalty);
            if (beam_hyps_min_normed_scores[wg_id] >=
                highest_attainable_score) {
              finished[wg_id] = true;
            }
          }
        }
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(batch_size * 32), sycl::range<1>(32)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

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
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      for (int32_t index = item.get_local_id(0); index < batch_size * beam_size;
           index += wg_size) {
        int32_t offset = time_step * batch_size * beam_size;
        int batch_idx = index / beam_size;

        beam_ids[offset + index] = top_beam_id[index];
        word_ids[offset + index] = top_word_id[index];
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

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
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
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
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(batch_size * 256), sycl::range<1>(256)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

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
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
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
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(batch_size * wg_size), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}