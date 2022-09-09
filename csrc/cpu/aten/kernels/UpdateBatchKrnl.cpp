#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>

#include <aten/UpdateBatch.h>

#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace torch_ipex::cpu::kernel;

#if defined(CPU_CAPABILITY_AVX512)

template <typename T>
inline void update_hidden_kernel(
    std::vector<int64_t> idx,
    at::Tensor hidden,
    at::Tensor hidden_prime) {
  auto* hidden_ptr = hidden.data_ptr<T>();
  auto* hidden_prime_ptr = hidden_prime.data_ptr<T>();

  int64_t idx_len = idx.size();
  int64_t ld = hidden.size(0);
  int64_t bs = hidden.size(1);
  int64_t feature_size = hidden.size(2);
  // TODO: at::parallel_for
  for (int64_t i = 0; i < ld; i++) {
    for (int64_t j = 0; j < idx_len; j++) {
      auto pos = i * bs * feature_size + idx[j] * feature_size;
      move_ker(&hidden_ptr[pos], &hidden_prime_ptr[pos], feature_size);
    }
  }
}

template <typename T>
inline void update_feature_kernel(
    at::Tensor x,
    at::Tensor f,
    const at::Tensor& time_idxs,
    int batch_size,
    int max_len) {
  auto* x_ptr = x.data_ptr<T>();
  auto* f_ptr = f.data_ptr<T>();
  int32_t* time_idxs_ptr = static_cast<int32_t*>(time_idxs.data_ptr());

  int64_t time_step = x.size(1);
  int64_t feature_size = x.size(2);

  at::parallel_for(0, batch_size, 16, [&](int64_t start, int64_t end) {
    for (int i = start; i < end; i++) {
      int fetch_time_idx = std::min(time_idxs_ptr[i], max_len - 1);

      // f is a view of x: f = x[:, 0, :], f.data_ptr() ==
      // x.data_ptr() x has been transposed: x.transpose(0, 1) shape of x: [64,
      // 545, 1024] (bs * t * feature) stride of x: [1024, 65536, 1]

      // shape of f: [64, 1024]
      // stride of f: [1024, 1]
      auto x_pos =
          fetch_time_idx * batch_size * feature_size + i * feature_size;
      auto f_pos = i * feature_size;
      move_ker(&f_ptr[f_pos], &x_ptr[x_pos], feature_size);
    }
  });
}

inline void label_index_put_kernel(
    at::Tensor label_tensor_out,
    at::Tensor label_to_put_out,
    const at::Tensor& label_col,
    at::Tensor label_for_next_loop_out,
    int64_t max_symbols,
    int64_t batch_size,
    int64_t max_len) {
  // label_tensor.index_put_([label_row, label_col.to(torch.int64)],
  // label_to_put, accumulate=True)
  int64_t* label_tensor_out_ptr =
      static_cast<int64_t*>(label_tensor_out.data_ptr());
  int64_t* label_to_put_out_ptr =
      static_cast<int64_t*>(label_to_put_out.data_ptr());

  int32_t* label_col_ptr = static_cast<int32_t*>(label_col.data_ptr());

  at::parallel_for(0, batch_size, 16, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      label_tensor_out_ptr[i * max_len * max_symbols + label_col_ptr[i]] +=
          label_to_put_out_ptr[i];
    }
  });

  // label_tensor.gather(1, label_col.to(torch.int64).unsqueeze(1))
  // TODO: merge with label_tensor_out_ptr
  int64_t* label_for_next_loop_out_ptr =
      static_cast<int64_t*>(label_for_next_loop_out.data_ptr());
  at::parallel_for(0, batch_size, 16, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      label_for_next_loop_out_ptr[i] =
          label_tensor_out_ptr[i * max_len * max_symbols + label_col_ptr[i]];
    }
  });
}

inline void update_hidden_idx_kernel(
    at::Tensor not_blank_out,
    at::Tensor hidden_0,
    at::Tensor hidden_1,
    const at::Tensor& hidden_prime_0,
    const at::Tensor& hidden_prime_1,
    int64_t batch_size) {
  std::vector<int64_t> idx;
  int32_t* not_blank_out_ptr = static_cast<int32_t*>(not_blank_out.data_ptr());
  // TODO: merge idx into update_hidden_kernel
  // Have data dependency here, shouldn't be parallelled
  for (int64_t i = 0; i < batch_size; i++) {
    if (not_blank_out_ptr[i] != 0)
      idx.push_back(i);
  }

  AT_ASSERTM(
      hidden_0.scalar_type() == hidden_1.scalar_type(),
      "hidden_0 and hidden_1 should be in same dtype.");
  AT_ASSERTM(
      (hidden_0.scalar_type() == at::kBFloat16 ||
       hidden_0.scalar_type() == at::kFloat),
      "only support hidden_0 to be float or bf16 tensor");

  if (hidden_0.scalar_type() == at::kBFloat16) {
    update_hidden_kernel<at::BFloat16>(idx, hidden_0, hidden_prime_0);
    update_hidden_kernel<at::BFloat16>(idx, hidden_1, hidden_prime_1);
  } else {
    update_hidden_kernel<float>(idx, hidden_0, hidden_prime_0);
    update_hidden_kernel<float>(idx, hidden_1, hidden_prime_1);
  }
}

inline void update_feature_idx_kernel(
    const at::Tensor& blankness_out,
    at::Tensor x,
    at::Tensor f,
    const at::Tensor& time_idxs,
    int64_t batch_size,
    int64_t max_len) {
  if (should_update_feature(blankness_out, (int32_t)batch_size)) {
    AT_ASSERTM(
        (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat),
        "only support x to be float or bf16 tensor");
    if (x.scalar_type() == at::kBFloat16) {
      update_feature_kernel<at::BFloat16>(
          x, f, time_idxs, (int32_t)batch_size, (int32_t)max_len);
    } else {
      update_feature_kernel<float>(
          x, f, time_idxs, (int32_t)batch_size, (int32_t)max_len);
    }
  }
}
#endif

/*
  rnnt_update_batch: used in the batched_decoder of RNN-T.
  Check if the generated symbol is valid,
  update the hidden state, the label for the next loop and the final label
  result, break when all the time_idx have been processed.

  k: index of the max prob, [batch_size], torch.int64
  out_lens: valid time step of the encoded feature, [batch_size], torch.int32
  label_col: column of the final label result to be updated, [batch_size],
  torch.int32 symbols_added: whether there are non_blank symbols added,
  [batch_size], torch.int32 time_idxs: the current time idx of feature to use,
  [batch_size], torch.int32 blankness_out: if the current symbol is blank_id,
  [batch_size], torch.int32 blankvec_out: current symbol is blank_id or larger
  than valid time step, [batch_size], torch.int32 not_blank_out: if the current
  symbol is not blank_id, [batch_size], torch.int32 label_to_put_out: the label
  to put into the final label tensor, [batch_size], torch.int64
  label_tensor_out: the final label tensor for a batch, [batch_size,
  max_len*self.max_symbols], torch.int64 label_for_next_loop_out: the label to
  use for the next loop, [batch_size], torch.int64

  hidden_0: the hx to be updated for next loop, [D∗num_layers, batch_size, 320],
  f32 or bf16 hidden_1: the cx to be updated for next loop, [D∗num_layers,
  batch_size, 320], f32 hidden_prime_0: the hx calculated for the current loop,
  [D∗num_layers, batch_size, 320], f32 or bf16 hidden_prime_1: the cx calculated
  for the current loop, [D∗num_layers, batch_size, 320], f32 x: the feature got
  from the encoder. dim 0 and dim1 of x has been transposed in the encoder,
  [batch_size, time_step, 1024], f32 or bf16 f: the feature of the corresponding
  time idx. [batch_size, 1, 1024],same dtype as x

  max_symbols: the max symbols to generate: 30
  blank_id: id for blank symbol: 28
  batch_size: the batch size of the input
  _SOS: the mark of the Start Of Sequence: -1
  max_len: the maximum of out_lens
*/

bool rnnt_update_batch_kernel_impl(
    const at::Tensor& k,
    const at::Tensor& out_lens,
    at::Tensor label_col,
    at::Tensor symbols_added,
    at::Tensor time_idxs,
    at::Tensor blankness_out,
    at::Tensor blankvec_out,
    at::Tensor not_blank_out,
    at::Tensor label_to_put_out,
    at::Tensor label_tensor_out,
    at::Tensor label_for_next_loop_out,
    at::Tensor hidden_0,
    at::Tensor hidden_1,
    const at::Tensor& hidden_prime_0,
    const at::Tensor& hidden_prime_1,
    at::Tensor x,
    at::Tensor f,
    int64_t max_symbols,
    int64_t blank_id,
    int64_t batch_size,
    int64_t _SOS,
    int64_t max_len) {
#if defined(CPU_CAPABILITY_AVX512)
  update_batch_kernel(
      k,
      out_lens,
      label_col,
      symbols_added,
      time_idxs,
      blankness_out,
      blankvec_out,
      not_blank_out,
      label_to_put_out,
      (int32_t)max_symbols,
      (int32_t)blank_id,
      (int32_t)batch_size,
      (int32_t)_SOS);

  // if blank_vec.nonzero().size(0) == batch_size:
  //     # all time_idxs processed, stop
  //     break
  if (all_time_idxs_processed_kernel(blankvec_out, batch_size))
    return BatchStatus::Finished;

  // label_tensor.index_put_([label_row, label_col.to(torch.int64)],
  // label_to_put, accumulate=True) label_tensor.gather(1,
  // label_col.to(torch.int64).unsqueeze(1))
  label_index_put_kernel(
      label_tensor_out,
      label_to_put_out,
      label_col,
      label_for_next_loop_out,
      max_symbols,
      batch_size,
      max_len);

  // idx = (not_blank).nonzero(as_tuple=True)[0]
  // hidden[0][:, idx, :] = hidden_prime[0][:, idx, :]
  // hidden[1][:, idx, :] = hidden_prime[1][:, idx, :]
  update_hidden_idx_kernel(
      not_blank_out,
      hidden_0,
      hidden_1,
      hidden_prime_0,
      hidden_prime_1,
      batch_size);

  // if blankness.nonzero().size(0) > 0:
  //     fetch_time_idxs = time_idxs.min(max_lens)
  //     f = x[label_row_list, fetch_time_idxs, :]
  update_feature_idx_kernel(
      blankness_out, x, f, time_idxs, batch_size, max_len);
  return BatchStatus::UnFinished;
#else
  // label_row = torch.tensor([i for i in range(batch_size)])
  std::vector<long> v;
  for (int64_t i = 0; i < batch_size; i++) {
    v.push_back(i);
  }
  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  at::Tensor label_row = torch::from_blob(v.data(), {batch_size}, opts);

  // blankness = k.eq(_blank_id)
  // time_idxs = time_idxs + blankness
  // blank_vec = time_idxs.ge(out_lens)
  blankness_out.copy_(k.eq(blank_id).to(at::ScalarType::Int));
  time_idxs.add_(blankness_out);
  blankvec_out.copy_(time_idxs.ge(out_lens));

  // symbols_added *= blankness.logical_not()
  // tmp_blank_vec = blank_vec.logical_or(blankness)
  symbols_added.mul_(blankness_out.logical_not());
  auto tmp_blank_vec = blankvec_out.logical_or(blankness_out);

  // not_blank = tmp_blank_vec.eq(0)
  // idx = (not_blank).nonzero(as_tuple=True)[0]
  // hidden_0[:, idx, :] = hidden_prime_0[:, idx, :]
  // hidden_1[:, idx, :] = hidden_prime_1[:, idx, :]
  not_blank_out.copy_(tmp_blank_vec.eq(false).to(at::ScalarType::Int));
  auto idx = not_blank_out.nonzero_numpy()[0];
  hidden_0.index_put_(
      {at::indexing::Slice(), idx, at::indexing::Slice()},
      hidden_prime_0.index(
          {at::indexing::Slice(), idx, at::indexing::Slice()}));
  hidden_1.index_put_(
      {at::indexing::Slice(), idx, at::indexing::Slice()},
      hidden_prime_1.index(
          {at::indexing::Slice(), idx, at::indexing::Slice()}));

  // label_col += not_blank
  // label_tensor.index_put_([label_row, label_col.to(torch.int64)],
  // (k-_SOS)*not_blank, accumulate=True)
  label_col.add_(not_blank_out);
  label_tensor_out.index_put_(
      {label_row, label_col.to(at::ScalarType::Long)},
      (k - _SOS) * tmp_blank_vec.eq(false),
      true);

  // symbols_added += not_blank
  // need_add = symbols_added.ge(max_symbols)
  // time_idxs += need_add
  // blankness.logical_or_(need_add)
  // symbols_added *= symbols_added.lt(max_symbols)
  symbols_added.add_(not_blank_out);
  time_idxs.add_(symbols_added.ge(max_symbols));
  blankness_out.logical_or_(
      symbols_added.ge(max_symbols).to(at::ScalarType::Int));
  symbols_added.mul_(symbols_added.lt(max_symbols));

  // max_lens = torch.tensor([max_len-1 for i in range(batch_size)],
  // dtype=torch.int64)
  std::vector<long> lens_value;
  for (int64_t i = 0; i < batch_size; i++) {
    lens_value.push_back(max_len - 1);
  }
  auto opts_ = torch::TensorOptions().dtype(torch::kInt64);
  at::Tensor max_lens =
      torch::from_blob(lens_value.data(), {batch_size}, opts_);

  // if blankness.nonzero().size(0) > 0:
  if (blankness_out.nonzero().size(0) > 0) {
    // fetch_time_idxs = time_idxs.min(max_lens)
    auto fetch_time_idxs = time_idxs.min(max_lens);

    // f = x[list(range(x.size(0))), fetch_time_idxs.to(torch.int64), :]
    f.copy_(x.index(
        {label_row,
         fetch_time_idxs.to(at::ScalarType::Long),
         at::indexing::Slice()}));
  }

  // label_for_next_loop = label_tensor.gather(1,
  // label_col.to(torch.int64).unsqueeze(1))
  label_for_next_loop_out.copy_(
      label_tensor_out
          .gather(1, label_col.to(at::ScalarType::Long).unsqueeze(1))
          .squeeze(1));

  // # all time_idxs processed, stop
  // finished = blank_vec.nonzero().size(0) == batch_size
  if (blankvec_out.nonzero().size(0) == batch_size) {
    return BatchStatus::Finished;
  } else {
    return BatchStatus::UnFinished;
  }
#endif
}

} // anonymous namespace

REGISTER_DISPATCH(
    rnnt_update_batch_kernel_stub,
    &rnnt_update_batch_kernel_impl);

} // namespace cpu
} // namespace torch_ipex