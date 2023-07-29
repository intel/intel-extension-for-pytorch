#include "BeamSearch.h"
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, int32_t MAX_K>
void beam_search_topk_kernel(
    const scalar_t* scores,
    scalar_t* tmp_scores,
    int64_t* tmp_idx,
    scalar_t* tmp_topk_scores,
    int64_t* tmp_topk_idx,
    int64_t end_id,
    int64_t vocab_size,
    int64_t beam_size,
    int64_t batch_size,
    int32_t num_wg_per_beam) {
  beam_search_topk_stage1<scalar_t, MAX_K * 2>(
      scores,
      tmp_scores,
      tmp_idx,
      end_id,
      vocab_size,
      beam_size,
      batch_size,
      num_wg_per_beam);

  beam_search_topk_stage2<scalar_t, MAX_K * 2>(
      tmp_scores,
      tmp_idx,
      tmp_topk_scores,
      tmp_topk_idx,
      vocab_size,
      beam_size,
      batch_size,
      num_wg_per_beam);
}

template <typename scalar_t>
void beam_search_topk_launch(
    const scalar_t* scores,
    scalar_t* tmp_scores,
    int64_t* tmp_idx,
    scalar_t* tmp_topk_scores,
    int64_t* tmp_topk_idx,
    int64_t end_id,
    int64_t vocab_size,
    int64_t beam_size,
    int64_t batch_size,
    int32_t num_wg_per_beam) {
#define CASE_K(K)                         \
  case K:                                 \
    beam_search_topk_kernel<scalar_t, K>( \
        scores,                           \
        tmp_scores,                       \
        tmp_idx,                          \
        tmp_topk_scores,                  \
        tmp_topk_idx,                     \
        end_id,                           \
        vocab_size,                       \
        beam_size,                        \
        batch_size,                       \
        num_wg_per_beam);                 \
    break;

  switch (beam_size) {
    CASE_K(4);
    CASE_K(8);
    CASE_K(16);
    default:
      throw std::runtime_error("beam size is not supported!");
  }
}

template <typename scalar_t, int32_t MAX_K>
void beam_seach_procss_impl(
    const scalar_t* tmp_score, // in
    const int64_t* tmp_idx, // in
    scalar_t* topk_score, // out
    int64_t* topk_token, // out
    int64_t* topk_beams, // out
    const int64_t pad_token_id,
    const int64_t eos_token_id,
    bool* finished,
    const float length_penalty,
    const int process_length, // beam_size x beam_size x 2
    const int64_t beam_size,
    const int64_t batch_size,
    const int64_t vocab_size,
    int64_t cur_len,
    bool early_stopping,
    int64_t max_in_seq_len,
    int64_t max_out_seq_len,
    int64_t* cadidate_num_beams, // [batch_size] number for eos candidates
    scalar_t* candidate_min_normed_scores, // [batch_size]
    scalar_t* cadidate_normed_scores, // [batch_size * 2 * beam_size], store
                                      // the norm scores for candidates
    int64_t* candidate_output_ids, // [batch_size * 2 * beam_size, max_seq],
                                   // cadidate output sentence
    int64_t* output_token_ids, // [max_seq, batch_size * 2 * beam_size],
                               // the out_buffer
    int64_t* output_beam_ids, // [max_seq, batch_size, beam_size]
    int64_t* candidate_sequence_lengths, // [batch_size * 2 * beam_size]
    scalar_t* candidate_score) { // [batch_size * 2 * beam_size]

  batch_topk_kernel<scalar_t, MAX_K * 2>(
      tmp_score,
      tmp_idx,
      topk_score,
      topk_token,
      topk_beams,
      pad_token_id,
      eos_token_id,
      finished,
      cadidate_num_beams,
      candidate_min_normed_scores,
      cadidate_normed_scores,
      candidate_output_ids,
      output_token_ids,
      output_beam_ids,
      candidate_sequence_lengths,
      candidate_score,
      length_penalty,
      process_length,
      beam_size,
      batch_size,
      vocab_size,
      cur_len,
      max_in_seq_len,
      max_out_seq_len,
      early_stopping);
}

// input: tmp_score [batch_size * beam_size * beam_size * 2]
//        tmp_idx
// output: topk_score [batch_size * beam_size]
//         topk_idx
template <typename scalar_t>
void beam_seach_procss(
    const scalar_t* tmp_score,
    const int64_t* tmp_idx,
    scalar_t* topk_score,
    int64_t* topk_token,
    int64_t* topk_beams,
    const int64_t pad_token_id,
    const int64_t eos_token_id,
    bool* finished,
    const float length_penalty,
    const int process_length,
    const int64_t beam_size,
    const int64_t batch_size,
    const int64_t vocab_size,
    int64_t cur_len,
    bool early_stopping,
    int64_t max_in_seq_len,
    int64_t max_out_seq_len,
    int64_t* cadidate_num_beams,
    scalar_t* candidate_min_normed_scores,
    scalar_t* cadidate_normed_scores,
    int64_t* candidate_output_ids,
    int64_t* output_token_ids,
    int64_t* output_beam_ids,
    int64_t* candidate_sequence_lengths,
    scalar_t* candidate_score) {
#define CASE_P(K)                        \
  case K:                                \
    beam_seach_procss_impl<scalar_t, K>( \
        tmp_score,                       \
        tmp_idx,                         \
        topk_score,                      \
        topk_token,                      \
        topk_beams,                      \
        pad_token_id,                    \
        eos_token_id,                    \
        finished,                        \
        length_penalty,                  \
        process_length,                  \
        beam_size,                       \
        batch_size,                      \
        vocab_size,                      \
        cur_len,                         \
        early_stopping,                  \
        max_in_seq_len,                  \
        max_out_seq_len,                 \
        cadidate_num_beams,              \
        candidate_min_normed_scores,     \
        cadidate_normed_scores,          \
        candidate_output_ids,            \
        output_token_ids,                \
        output_beam_ids,                 \
        candidate_sequence_lengths,      \
        candidate_score);                \
    break;

  switch (beam_size) {
    CASE_P(4);
    CASE_P(8);
    CASE_P(16);
    default:
      throw std::runtime_error("beam size is not supported!");
  }
}

// input: scores [batch_size*beam_size, vocab_size]
// output: top result
//        top_score [batch_size * beam_size]
//        top_token [batch_size * beam_size]
//        top_idx   [batch_size * beam_size]
std::tuple<Tensor, Tensor, Tensor> beam_search_topk(
    const Tensor& logits_score,
    Tensor& finished,
    int64_t pad_token_id,
    int64_t eos_token_id,
    double length_penalty,
    const int64_t beam_size,
    const int64_t batch_size,
    const int64_t vocab_size,
    int64_t cur_len,
    bool early_stopping,
    const int64_t max_in_seq_len,
    const int64_t max_out_seq_len,
    Tensor& output_token_ids,
    Tensor& output_beam_ids,
    Tensor& cadidate_num_beams,
    Tensor& cadidate_normed_scores,
    Tensor& candidate_min_normed_scores,
    Tensor& candidate_output_ids,
    Tensor& candidate_sequence_lengths,
    Tensor& candidate_score) {
  RECORD_FUNCTION(
      "beam_search_topk",
      std::vector<c10::IValue>(
          {logits_score,
           finished,
           output_token_ids,
           output_beam_ids,
           cadidate_num_beams,
           cadidate_normed_scores,
           candidate_min_normed_scores,
           candidate_output_ids,
           candidate_sequence_lengths,
           candidate_score}));
  const int32_t num_wg_per_beam = 128;
  const int32_t tmp_output_len =
      2 * beam_size * num_wg_per_beam * beam_size * batch_size;
  const int32_t topk_len = 2 * beam_size * beam_size * batch_size;
  const int32_t process_length = 2 * beam_size * beam_size;
  // Tensor tmp_log_score = at::empty(logits_score.sizes(),
  // logits_score.options());
  Tensor tmp_log_val = at::empty(tmp_output_len, logits_score.options());
  Tensor tmp_log_idx =
      at::empty(tmp_output_len, logits_score.options().dtype(at::kLong));

  Tensor tmp_topk_val = at::empty(topk_len, logits_score.options());
  Tensor tmp_topk_idx =
      at::empty(topk_len, logits_score.options().dtype(at::kLong));

  Tensor topk_val = at::empty(beam_size * batch_size, logits_score.options());
  Tensor topk_token = at::empty(
      beam_size * batch_size, logits_score.options().dtype(at::kLong));
  Tensor topk_idx = at::empty(
      beam_size * batch_size, logits_score.options().dtype(at::kLong));

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      logits_score.scalar_type(),
      "beam_search_topk_launch",
      [&]() {
        beam_search_topk_launch<scalar_t>(
            logits_score.data_ptr<scalar_t>(),
            tmp_log_val.data_ptr<scalar_t>(),
            tmp_log_idx.data_ptr<int64_t>(),
            tmp_topk_val.data_ptr<scalar_t>(),
            tmp_topk_idx.data_ptr<int64_t>(),
            eos_token_id,
            vocab_size,
            beam_size,
            batch_size,
            num_wg_per_beam);
      });

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      logits_score.scalar_type(),
      "beam_search_process",
      [&]() {
        beam_seach_procss<scalar_t>(
            tmp_topk_val.data_ptr<scalar_t>(),
            tmp_topk_idx.data_ptr<int64_t>(),
            topk_val.data_ptr<scalar_t>(),
            topk_token.data_ptr<int64_t>(),
            topk_idx.data_ptr<int64_t>(),
            pad_token_id,
            eos_token_id,
            finished.data_ptr<bool>(),
            (float)length_penalty,
            process_length,
            beam_size,
            batch_size,
            vocab_size,
            cur_len,
            early_stopping,
            max_in_seq_len,
            max_out_seq_len,
            cadidate_num_beams.data_ptr<int64_t>(),
            candidate_min_normed_scores.data_ptr<scalar_t>(),
            cadidate_normed_scores.data_ptr<scalar_t>(),
            candidate_output_ids.data_ptr<int64_t>(),
            output_token_ids.data_ptr<int64_t>(),
            output_beam_ids.data_ptr<int64_t>(),
            candidate_sequence_lengths.data_ptr<int64_t>(),
            candidate_score.data_ptr<scalar_t>());
      });

  return std::tuple<Tensor, Tensor, Tensor>(topk_val, topk_token, topk_idx);
}

void update_output_indices(
    const Tensor& top_beam_id,
    const Tensor& top_token_id,
    Tensor& output_beam_ids,
    Tensor& output_token_ids,
    Tensor& finished,
    const int64_t time_step,
    const int64_t batch_size,
    const int64_t beam_size) {
  RECORD_FUNCTION(
      "beam_search_topk",
      std::vector<c10::IValue>(
          {top_beam_id,
           top_token_id,
           output_beam_ids,
           output_token_ids,
           finished}));
  update_token(
      top_beam_id.data_ptr<int64_t>(),
      top_token_id.data_ptr<int64_t>(),
      output_beam_ids.data_ptr<int64_t>(),
      output_token_ids.data_ptr<int64_t>(),
      finished.data_ptr<bool>(),
      time_step,
      batch_size,
      beam_size);
}

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
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      int32_t time_step =
          item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
      int32_t sentence_id =
          item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

      int32_t beam_id = sentence_id % beam_size;
      int32_t batch_id = sentence_id / beam_size;
      int32_t offset = num_step * batch_size * beam_size;

      if (sentence_id < batch_size * beam_size && time_step < num_step) {
        const scalar_t src_beam = beam_ids[sentence_id];
        // const scalar_t src_beam =
        //     (scalar_t)beam_ids[sentence_id] - batch_id * beam_size;
        // fix for reference beam search
        const int32_t src_offset = batch_size * beam_size * time_step +
            batch_id * beam_size + src_beam;
        const int32_t out_offset =
            batch_size * beam_size * time_step + batch_id * beam_size + beam_id;

        out_cache_indices[out_offset] =
            (time_step == step) ? beam_id : src_cache_indices[src_offset];
      }
    };

    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(wg_number * wg_size, batch_size * beam_size),
            sycl::range<2>(wg_size, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor update_beam_indices_for_cache(
    const Tensor& src_cache_indices,
    const Tensor& beam_ids,
    int64_t beam_size,
    int64_t batch_size) {
  Tensor out_cache_indices = at::empty(
      {src_cache_indices.size(0) + 1, src_cache_indices.size(1)},
      src_cache_indices.options());
  RECORD_FUNCTION(
      "update_beam_indices_for_cache",
      std::vector<c10::IValue>({src_cache_indices, beam_ids}));
  const int step = src_cache_indices.size(0);
  IPEX_DISPATCH_INTEGRAL_TYPES(
      src_cache_indices.scalar_type(), "update_beam_indices_for_cache", [&]() {
        update_beam_indices_kernel(
            src_cache_indices.data_ptr<scalar_t>(),
            out_cache_indices.data_ptr<scalar_t>(),
            beam_ids.data_ptr<int64_t>(),
            step,
            beam_size,
            batch_size);
      });
  return out_cache_indices;
}

Tensor beam_search_finalize(
    Tensor& candidate_num_beams,
    Tensor& candidate_sequence_lengths,
    Tensor& candidate_output_ids,
    Tensor& candidate_score,
    Tensor& candidate_normed_scores,
    Tensor& output_beam_ids,
    Tensor& output_token_ids,
    Tensor& top_score,
    Tensor& finished,
    double length_penalty,
    const int64_t max_in_seq_len,
    const int64_t max_out_seq_len,
    const int64_t batch_size,
    const int64_t beam_size,
    const int64_t out_sentence_number,
    const int64_t cur_len,
    const int64_t pad_token_id) {
  RECORD_FUNCTION(
      "beam_search_finalize",
      std::vector<c10::IValue>(
          {candidate_num_beams,
           candidate_sequence_lengths,
           candidate_output_ids,
           candidate_score,
           candidate_normed_scores,
           output_beam_ids,
           output_token_ids,
           top_score,
           finished}));
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      candidate_score.scalar_type(),
      "insert_to_candidate_list",
      [&]() {
        insert_to_candidate_list<scalar_t>(
            candidate_num_beams.data_ptr<int64_t>(),
            candidate_sequence_lengths.data_ptr<int64_t>(),
            output_token_ids.data_ptr<int64_t>(),
            candidate_output_ids.data_ptr<int64_t>(),
            output_beam_ids.data_ptr<int64_t>(),
            candidate_score.data_ptr<scalar_t>(),
            candidate_normed_scores.data_ptr<scalar_t>(),
            top_score.data_ptr<scalar_t>(),
            finished.data_ptr<bool>(),
            (float)length_penalty,
            max_in_seq_len,
            max_out_seq_len,
            batch_size,
            beam_size,
            cur_len);
      });

  Tensor output_ids = at::empty(
      {batch_size * out_sentence_number, max_in_seq_len + max_out_seq_len},
      candidate_output_ids.options());
  Tensor sequence_length = at::empty(
      {batch_size, out_sentence_number}, candidate_output_ids.options());

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      candidate_normed_scores.scalar_type(),
      "finalize",
      [&]() {
        finalize<scalar_t>(
            output_ids.data_ptr<int64_t>(),
            sequence_length.data_ptr<int64_t>(),
            candidate_output_ids.data_ptr<int64_t>(),
            candidate_sequence_lengths.data_ptr<int64_t>(),
            candidate_normed_scores.data_ptr<scalar_t>(),
            candidate_num_beams.data_ptr<int64_t>(),
            beam_size,
            batch_size,
            max_in_seq_len,
            max_out_seq_len,
            out_sentence_number,
            pad_token_id);
      });
  return output_ids;
}

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
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int32_t wi_id = item.get_local_id(0);
      int32_t wg_id = item.get_group(0);

      for (int32_t index = wi_id; index < input_len; index += wg_size) {
        output_ids[wg_id * output_len + index] =
            input_ids[(wg_id / out_beams) * beam_size * input_len + index];
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(wg_size * seq_num), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor& update_output_sequence(
    const Tensor& input_ids,
    Tensor& output_ids,
    const int64_t batch_size) {
  RECORD_FUNCTION(
      "beam_search_finalize",
      std::vector<c10::IValue>({input_ids, output_ids}));
  int32_t input_len = input_ids.size(1);
  int32_t output_len = output_ids.size(1);
  int32_t seq_num = output_ids.size(0);
  int32_t out_beams = seq_num / batch_size;
  int32_t beam_size = input_ids.size(0) / batch_size;

  IPEX_DISPATCH_INTEGRAL_TYPES(
      input_ids.scalar_type(), "update_output_sequence", [&]() {
        copy_input_to_output<scalar_t>(
            input_ids.data_ptr<scalar_t>(),
            output_ids.data_ptr<scalar_t>(),
            input_len,
            output_len,
            seq_num,
            beam_size,
            out_beams);
      });
  return output_ids;
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "update_beam_indices_for_cache",
      update_beam_indices_for_cache,
      c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "beam_search_topk", beam_search_topk, c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "update_output_indices", update_output_indices, c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "beam_search_finalize", beam_search_finalize, c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "update_output_sequence", update_output_sequence, c10::DispatchKey::XPU);
}

} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at