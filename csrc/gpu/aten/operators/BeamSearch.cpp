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
  impl::beam_search_topk_stage1<scalar_t, MAX_K * 2>(
      scores,
      tmp_scores,
      tmp_idx,
      end_id,
      vocab_size,
      beam_size,
      batch_size,
      num_wg_per_beam);

  impl::beam_search_topk_stage2<scalar_t, MAX_K * 2>(
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
#define CASE_K(K, MAX_K)                      \
  case K ... MAX_K:                           \
    beam_search_topk_kernel<scalar_t, MAX_K>( \
        scores,                               \
        tmp_scores,                           \
        tmp_idx,                              \
        tmp_topk_scores,                      \
        tmp_topk_idx,                         \
        end_id,                               \
        vocab_size,                           \
        beam_size,                            \
        batch_size,                           \
        num_wg_per_beam);                     \
    break;

  switch (beam_size) {
    CASE_K(1, 4);
    CASE_K(5, 8);
    CASE_K(9, 16);
    CASE_K(17, 32);
    // FIXME: HW limitation for scratch space. Lowering to SIMD16 also could fix
    // scratch space area is for register spill/fill.
    // CASE_K(33, 64);
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
    int64_t* candidate_num_beams, // [batch_size] number for eos candidates
    scalar_t* candidate_min_normed_scores, // [batch_size]
    scalar_t* candidate_normed_scores, // [batch_size * 2 * beam_size], store
                                       // the norm scores for candidates
    int64_t* candidate_output_ids, // [batch_size * 2 * beam_size, max_seq],
                                   // candidate output sentence
    int64_t* output_token_ids, // [max_seq, batch_size * 2 * beam_size],
                               // the out_buffer
    int64_t* output_beam_ids, // [max_seq, batch_size, beam_size]
    int64_t* candidate_sequence_lengths, // [batch_size * 2 * beam_size]
    scalar_t* candidate_score) { // [batch_size * 2 * beam_size]

  impl::batch_topk_kernel<scalar_t, MAX_K * 2>(
      tmp_score,
      tmp_idx,
      topk_score,
      topk_token,
      topk_beams,
      pad_token_id,
      eos_token_id,
      finished,
      candidate_num_beams,
      candidate_min_normed_scores,
      candidate_normed_scores,
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
    int64_t* candidate_num_beams,
    scalar_t* candidate_min_normed_scores,
    scalar_t* candidate_normed_scores,
    int64_t* candidate_output_ids,
    int64_t* output_token_ids,
    int64_t* output_beam_ids,
    int64_t* candidate_sequence_lengths,
    scalar_t* candidate_score) {
#define CASE_P(K, MAX_K)                     \
  case K ... MAX_K:                          \
    beam_seach_procss_impl<scalar_t, MAX_K>( \
        tmp_score,                           \
        tmp_idx,                             \
        topk_score,                          \
        topk_token,                          \
        topk_beams,                          \
        pad_token_id,                        \
        eos_token_id,                        \
        finished,                            \
        length_penalty,                      \
        process_length,                      \
        beam_size,                           \
        batch_size,                          \
        vocab_size,                          \
        cur_len,                             \
        early_stopping,                      \
        max_in_seq_len,                      \
        max_out_seq_len,                     \
        candidate_num_beams,                 \
        candidate_min_normed_scores,         \
        candidate_normed_scores,             \
        candidate_output_ids,                \
        output_token_ids,                    \
        output_beam_ids,                     \
        candidate_sequence_lengths,          \
        candidate_score);                    \
    break;

  switch (beam_size) {
    CASE_P(1, 4);
    CASE_P(5, 8);
    CASE_P(9, 16);
    CASE_P(17, 32);
    // CASE_P(33, 64);
    default:
      throw std::runtime_error("beam size is not supported!");
  }
}

// Select top high score token for each batch, beam_size * vocab_size number
// scores in total, then pick beam_size top score. If the selected token is eos,
// move the sentence to candidate pool.
// input: logits_score [batch_size*beam_size, vocab_size]
// output: top result
//        top_score [batch_size * beam_size]
//        top_token [batch_size * beam_size] token id
//        top_idx   [batch_size * beam_size] beam id
// candidate_xxx are variables to maintain the candidate pool
std::tuple<Tensor, Tensor, Tensor> beam_search_topk(
    const Tensor& logits_score, // in: all token scores
    Tensor& finished, // mark decode finishes
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
    Tensor& output_token_ids, // store token ids for each time step
    Tensor& output_beam_ids, // store beam ids for each time step
    Tensor& candidate_num_beams, // sentence number for candidate pool
    Tensor& candidate_normed_scores,
    Tensor& candidate_min_normed_scores,
    Tensor& candidate_output_ids, // sentences with eos (decode done)
    Tensor& candidate_sequence_lengths,
    Tensor& candidate_score) {
  DeviceId curDevID = at::xpu::current_device();
  int32_t eu_count = dpcppGpuEuCount();
  int32_t num_wg_per_beam;
  if (!Settings::I().has_2d_block_array(curDevID)) {
    // In case of OOM in some platforms
    num_wg_per_beam =
        std::min(CeilDiv(eu_count, (int32_t)(batch_size * beam_size)), 4);
  } else {
    num_wg_per_beam = CeilDiv(eu_count, (int32_t)(batch_size * beam_size));
  }

  const int32_t tmp_output_len =
      2 * beam_size * num_wg_per_beam * beam_size * batch_size;
  const int32_t topk_len = 2 * beam_size * beam_size * batch_size;
  const int32_t process_length = 2 * beam_size * beam_size;

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
            candidate_num_beams.data_ptr<int64_t>(),
            candidate_min_normed_scores.data_ptr<scalar_t>(),
            candidate_normed_scores.data_ptr<scalar_t>(),
            candidate_output_ids.data_ptr<int64_t>(),
            output_token_ids.data_ptr<int64_t>(),
            output_beam_ids.data_ptr<int64_t>(),
            candidate_sequence_lengths.data_ptr<int64_t>(),
            candidate_score.data_ptr<scalar_t>());
      });

  return std::tuple<Tensor, Tensor, Tensor>(topk_val, topk_token, topk_idx);
}

// Add current time step top tokens to global pool.
// global pool maintains high score token, non-eos.
void update_output_indices(
    const Tensor& top_beam_id, // current time step beam id
    const Tensor& top_token_id, // current time step token id
    Tensor& output_beam_ids, // global beam id
    Tensor& output_token_ids, // global token id
    Tensor& finished,
    const int64_t time_step,
    const int64_t batch_size,
    const int64_t beam_size) {
  impl::update_token(
      top_beam_id.data_ptr<int64_t>(),
      top_token_id.data_ptr<int64_t>(),
      output_beam_ids.data_ptr<int64_t>(),
      output_token_ids.data_ptr<int64_t>(),
      finished.data_ptr<bool>(),
      time_step,
      batch_size,
      beam_size);
}

// Reorder cached beam indices according to current time step beam index,
// and add current time step beam index
Tensor update_beam_indices_for_cache(
    const Tensor& src_cache_indices,
    const Tensor& beam_ids,
    int64_t beam_size,
    int64_t batch_size) {
  Tensor out_cache_indices = at::empty(
      {src_cache_indices.size(0) + 1, src_cache_indices.size(1)},
      src_cache_indices.options());
  const int step = src_cache_indices.size(0);
  IPEX_DISPATCH_INTEGRAL_TYPES(
      src_cache_indices.scalar_type(), "update_beam_indices_for_cache", [&]() {
        impl::update_beam_indices_kernel(
            src_cache_indices.data_ptr<scalar_t>(),
            out_cache_indices.data_ptr<scalar_t>(),
            beam_ids.data_ptr<int64_t>(),
            step,
            beam_size,
            batch_size);
      });
  return out_cache_indices;
}

Tensor update_native_beam_indices_for_cache(
    const Tensor& src_cache_indices,
    const Tensor& beam_ids,
    int64_t beam_size,
    int64_t batch_size) {
  Tensor out_cache_indices = at::empty(
      {src_cache_indices.size(0) + 1, src_cache_indices.size(1)},
      src_cache_indices.options());
  const int step = src_cache_indices.size(0);
  IPEX_DISPATCH_INTEGRAL_TYPES(
      src_cache_indices.scalar_type(),
      "update_native_beam_indices_for_cache",
      [&]() {
        impl::update_native_beam_indices_kernel(
            src_cache_indices.data_ptr<scalar_t>(),
            out_cache_indices.data_ptr<scalar_t>(),
            beam_ids.data_ptr<int64_t>(),
            step,
            beam_size,
            batch_size);
      });
  return out_cache_indices;
}

// Decoding stops when meet the stop condition. Pick final high score sentence
// as beam search output. When decoding ends, global/candidate pool maintain
// high score tokens for non-eos/eos sentences. Merge two lists and select top
// score sentence.
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
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      candidate_score.scalar_type(),
      "insert_to_candidate_list",
      [&]() {
        impl::insert_to_candidate_list<scalar_t>(
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
        impl::finalize<scalar_t>(
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

// To align huggingface, beam search outputs is input prompt + decode sentence.
// Copy input prompt to output buffer here.
Tensor& update_output_sequence(
    const Tensor& input_ids,
    Tensor& output_ids,
    const int64_t batch_size) {
  int32_t input_len = input_ids.size(1);
  int32_t output_len = output_ids.size(1);
  int32_t seq_num = output_ids.size(0);
  int32_t out_beams = seq_num / batch_size;
  int32_t beam_size = input_ids.size(0) / batch_size;

  IPEX_DISPATCH_INTEGRAL_TYPES(
      input_ids.scalar_type(), "update_output_sequence", [&]() {
        impl::copy_input_to_output<scalar_t>(
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
      "update_native_beam_indices_for_cache",
      update_native_beam_indices_for_cache,
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