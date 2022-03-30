#include <torch/extension.h>

#include "WeightPack.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/utils/rw_lock.h"
#include "utils/utils.h"

namespace torch_ipex {
namespace cpu {

bool is_transposed_2d(const at::Tensor& tensor) {
  return (
      tensor.ndimension() == 2 && tensor.stride(0) == 1 &&
      tensor.stride(1) == tensor.size(0));
}

namespace {

using weakref_type =
    c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using val_blocked = std::tuple<weakref_type, ideep::tensor>;
std::unordered_map<c10::TensorImpl*, val_blocked> cached_weights;

using map_iter =
    std::unordered_map<c10::TensorImpl*, val_blocked>::const_iterator;

torch_ipex::ReadWriteMutex rwmutex;

ideep::tensor read_cached_weights(const at::Tensor& weight) {
  torch_ipex::UniqueReadLock<torch_ipex::ReadWriteMutex> lock(rwmutex);
  ideep::tensor cached_weight;
  auto it = cached_weights.find(weight.unsafeGetTensorImpl());
  if (it != cached_weights.end()) {
    cached_weight = std::get<1>(it->second);
  }
  return cached_weight;
}

void write_cached_weights(const at::Tensor& weight, ideep::tensor& result) {
  torch_ipex::UniqueWriteLock<torch_ipex::ReadWriteMutex> lock(rwmutex);
  cached_weights.emplace(
      weight.unsafeGetTensorImpl(),
      val_blocked{weakref_type(weight.getIntrusivePtr()), result});
}

} // namespace

// Get the convolution's expected ideep weight tensor desc.
ideep::tensor::desc get_conv_expected_weights_desc(
    const ideep::tensor::dims& weights_dims,
    ideep::tensor::data_type w_dtype,
    const ideep::tensor::dims& strides,
    const ideep::tensor::dims& padding_l,
    const ideep::tensor::dims& padding_r,
    const ideep::tensor::dims& dilates,
    int groups,
    bool channels_last,
    ideep::algorithm aalgorithm,
    ideep::data_type x_dtype,
    const ideep::dims& src_dims,
    const ideep::attr_t& attr) {
  if (channels_last) {
    return ideep::convolution_forward::expected_weights_desc<true>(
        weights_dims,
        w_dtype,
        strides,
        padding_l,
        padding_r,
        dilates,
        groups,
        aalgorithm,
        ideep::prop_kind::forward,
        x_dtype,
        src_dims,
        attr);
  } else {
    return ideep::convolution_forward::expected_weights_desc<false>(
        weights_dims,
        w_dtype,
        strides,
        padding_l,
        padding_r,
        dilates,
        groups,
        aalgorithm,
        ideep::prop_kind::forward,
        x_dtype,
        src_dims,
        attr);
  }
}

bool is_packed(const at::Tensor& weight) {
  auto cached_weight = read_cached_weights(weight);
  return !cached_weight.is_empty();
}

std::tuple<ideep::tensor, ideep::tensor> get_lstm_packed_weight(
    const at::Tensor& weight_ih,
    const at::Tensor& weight_hh,
    int64_t input_size,
    int64_t num_gates,
    int64_t hidden_size,
    const ideep::dims& output_sizes,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& bias,
    const bool reverse,
    const bool train) {
  // TODO: This is a workaround. In training, weight prepack is not enabled
  // Will remove this when weight prepack is enabled.
  if (train) {
    auto w1 = itensor_view_from_dense(
        weight_ih,
        {{1, 1, input_size, num_gates, hidden_size},
         get_mkldnn_dtype(weight_ih.scalar_type()),
         ideep::format_tag::ldgoi});
    auto w2 = itensor_view_from_dense(
        weight_hh,
        {{1, 1, hidden_size, num_gates, hidden_size},
         get_mkldnn_dtype(weight_hh.scalar_type()),
         ideep::format_tag::ldgoi});
    return std::make_tuple(w1, w2);
  }
  auto cached_weight_ih = read_cached_weights(weight_ih);
  auto cached_weight_hh = read_cached_weights(weight_hh);
  bool all_in_cache =
      !cached_weight_ih.is_empty() && !cached_weight_hh.is_empty();
  bool all_miss = cached_weight_ih.is_empty() && cached_weight_hh.is_empty();
  TORCH_CHECK(
      all_in_cache || all_miss,
      "both of the weights of LSTM should be "
      "cached or neither should be cached");

  if (cached_weight_ih.is_empty()) {
    auto w1 = itensor_view_from_dense(
        weight_ih,
        {{1, 1, input_size, num_gates, hidden_size},
         get_mkldnn_dtype(weight_ih.scalar_type()),
         ideep::format_tag::ldgoi});
    auto w2 = itensor_view_from_dense(
        weight_hh,
        {{1, 1, hidden_size, num_gates, hidden_size},
         get_mkldnn_dtype(weight_hh.scalar_type()),
         ideep::format_tag::ldgoi});

    ideep::tensor::desc packed_desc_ih, packed_desc_hh;
    if (train) {
      std::tie(packed_desc_ih, packed_desc_hh) =
          ideep::lstm_forward_training::expected_weights_desc(
              output_sizes,
              src_layer,
              src_iter,
              src_iter_c,
              w1,
              w2,
              bias,
              reverse);
    } else {
      std::tie(packed_desc_ih, packed_desc_hh) =
          ideep::lstm_forward_inference::expected_weights_desc(
              output_sizes,
              src_layer,
              src_iter,
              src_iter_c,
              w1,
              w2,
              bias,
              reverse);
    }

    // Don't pack when the weight is of rnn_packed format
    // When the weight is of rnn_packed format, if the seq_lens of
    // the input changes, the format of weight also changes.
    // oneDNN does not support reorder from rnn_packed back to public format.
    // LSTM based on BRGEMM kernel (on AVX512 and newest ISAs) will use blocked
    // format for weight of LSTM, which won't change when the input seq_lens
    // changes.
    if (packed_desc_ih.is_rnn_packed() || packed_desc_hh.is_rnn_packed()) {
      return std::make_tuple(w1, w2);
    }
    cached_weight_ih.init(packed_desc_ih);
    cached_weight_hh.init(packed_desc_hh);

    cached_weight_ih.feed_from(w1);
    cached_weight_hh.feed_from(w2);

    write_cached_weights(weight_ih, cached_weight_ih);
    write_cached_weights(weight_hh, cached_weight_hh);
  }
  return std::make_tuple(cached_weight_ih, cached_weight_hh);
}

ideep::tensor::desc get_conv_transpose2d_expected_weights_desc(
    const ideep::tensor::dims& weights_dims,
    ideep::tensor::data_type w_dtype,
    const ideep::tensor::dims& strides,
    const ideep::tensor::dims& padding_l,
    const ideep::tensor::dims& padding_r,
    const ideep::tensor::dims& dilates,
    int groups,
    bool channels_last,
    ideep::algorithm aalgorithm,
    ideep::data_type x_dtype,
    const ideep::dims& src_dims,
    const ideep::attr_t& attr) {
  if (channels_last) {
    return ideep::convolution_transpose_forward::expected_weights_desc<true>(
        weights_dims,
        w_dtype,
        strides,
        padding_l,
        padding_r,
        dilates,
        groups,
        aalgorithm,
        ideep::prop_kind::forward,
        src_dims,
        attr);
  } else {
    return ideep::convolution_transpose_forward::expected_weights_desc<false>(
        weights_dims,
        w_dtype,
        strides,
        padding_l,
        padding_r,
        dilates,
        groups,
        aalgorithm,
        ideep::prop_kind::forward,
        src_dims,
        attr);
  }
}

void sync_master_weight_to_bf16(
    const at::Tensor master_weight,
    at::Tensor& bf16_weight) {
  TORCH_CHECK(
      master_weight.sizes() == bf16_weight.sizes(),
      "expected master weight has same sizes with bf16 weight");
  TORCH_CHECK(
      master_weight.scalar_type() == at::kFloat &&
          bf16_weight.scalar_type() == at::kBFloat16,
      "expected master weght has same dims with bf16 weight");
  auto w_master = itensor_view_from_dense(master_weight);
  auto w_bf16 = itensor_view_from_dense(bf16_weight);
  w_bf16.feed_from(w_master);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "sync_master_weight_to_bf16(Tensor master_weight, Tensor bf16_weight) "
      "-> ()",
      torch_ipex::cpu::sync_master_weight_to_bf16);
}

} // namespace
