#include <torch/all.h>

#include "WeightPack.h"
#include "utils/rw_lock.h"
#include "utils/utils.h"

namespace torch_ipex {
namespace cpu {

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

bool is_packed(const at::Tensor& weight) {
  auto cached_weight = read_cached_weights(weight);
  return !cached_weight.is_empty();
}

#define LSTM_PACKED_WEIGHT(TYPE)                     \
  lstm_packed_weight<LstmInferenceWeightDesc<TYPE>>( \
      weight_ih,                                     \
      weight_hh,                                     \
      input_size,                                    \
      num_gates,                                     \
      hidden_size,                                   \
      output_sizes,                                  \
      src_layer,                                     \
      src_iter,                                      \
      src_iter_c,                                    \
      bias,                                          \
      reverse,                                       \
      quantizedLstmParams);

std::tuple<ideep::tensor, ideep::tensor> CommonLstmWeightDesc::
    get_and_save_lstm_packed_weight() {
  ideep::tensor cached_weight_ih, cached_weight_hh;
  // Don't pack when the weight is of opaque format (rnn_packed format).
  // When the weight is of rnn_packed format, if the seq_lens of
  // the input changes, the format of weight also changes.
  // oneDNN does not support reorder from rnn_packed back to public
  // format. LSTM based on BRGEMM kernel (on AVX512 and newest ISAs) will
  // use blocked format for weight of LSTM, which won't change when the
  // input seq_lens changes.
  if (packed_desc_ih_.is_opaque() || packed_desc_hh_.is_opaque()) {
    return std::make_tuple(w1_src_, w2_src_);
  }

  cached_weight_ih = w1_src_.reorder_if_differ_in(packed_desc_ih_, op_attr_);
  cached_weight_hh = w2_src_.reorder_if_differ_in(packed_desc_hh_, op_attr_);
  write_cached_weights(weight_ih_, cached_weight_ih);
  write_cached_weights(weight_hh_, cached_weight_hh);
  return std::make_tuple(cached_weight_ih, cached_weight_hh);
}

void LstmInferenceWeightDesc<LstmDtype::Float>::set_expected_weights_desc() {
  std::tie(packed_desc_ih_, packed_desc_hh_) =
      ideep::lstm_forward_inference::expected_weights_desc(
          output_sizes_,
          src_layer_,
          src_iter_,
          src_iter_c_,
          w1_src_,
          w2_src_,
          bias_,
          reverse_);
}

void LstmInferenceWeightDesc<LstmDtype::Quantized>::initialize_weight_src() {
  auto w1 = itensor_view_from_dense(
      weight_ih_,
      {{1, 1, input_size_, num_gates_, hidden_size_},
       get_mkldnn_dtype(weight_ih_.scalar_type()),
       ideep::format_tag::ldgoi});
  auto w2 = itensor_view_from_dense(
      weight_hh_,
      {{1, 1, hidden_size_, num_gates_, hidden_size_},
       get_mkldnn_dtype(weight_hh_.scalar_type()),
       ideep::format_tag::ldgoi});

  // The ideep tensor w1 and w2 which is a view of the input weight is in
  // ldgoi format. When querying the format of oneDNN:
  // - for int8, w1_src and w2_src should be in abcde (ldigo) format
  // - for fp32 and bf16, w1_src and w2_src could be in abdec (ldgoi) format
  ideep::tensor w1_src;
  ideep::tensor w2_src;

  ideep::tensor::desc w1_src_desc = {
      {1, 1, input_size_, num_gates_, hidden_size_},
      get_mkldnn_dtype(weight_ih_.scalar_type()),
      ideep::format_tag::ldigo};
  ideep::tensor::desc w2_src_desc = {
      {1, 1, hidden_size_, num_gates_, hidden_size_},
      get_mkldnn_dtype(weight_hh_.scalar_type()),
      ideep::format_tag::ldigo};

  w1_src = ideep::tensor({w1_src_desc});
  w2_src = ideep::tensor({w2_src_desc});

  w1.reorder_to(w1_src);
  w2.reorder_to(w2_src);

  w1_src_ = w1_src;
  w2_src_ = w2_src;
}

void LstmInferenceWeightDesc<
    LstmDtype::Quantized>::set_expected_weights_desc() {
  std::tie(packed_desc_ih_, packed_desc_hh_) =
      ideep::lstm_forward_inference::expected_weights_desc(
          output_sizes_,
          src_layer_,
          src_iter_,
          src_iter_c_,
          w1_src_,
          w2_src_,
          bias_,
          reverse_,
          scale_,
          zp_,
          weights_scale_mask_,
          weights_scales_);
}

template <typename lstm_param>
std::tuple<ideep::tensor, ideep::tensor> lstm_packed_weight(
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
    const QuantizedLstmParams& quantizedLstmParams) {
  auto cached_weight_ih = read_cached_weights(weight_ih);
  auto cached_weight_hh = read_cached_weights(weight_hh);
  bool all_in_cache =
      !cached_weight_ih.is_empty() && !cached_weight_hh.is_empty();
  bool all_miss = cached_weight_ih.is_empty() && cached_weight_hh.is_empty();
  TORCH_CHECK(
      all_in_cache || all_miss,
      "both of the weights of LSTM should be "
      "cached or neither should be cached");

  if (!cached_weight_ih.is_empty()) {
    return std::make_tuple(cached_weight_ih, cached_weight_hh);
  }

  lstm_param inference_weight_desc(
      {weight_ih,
       weight_hh,
       input_size,
       num_gates,
       hidden_size,
       output_sizes,
       src_layer,
       src_iter,
       src_iter_c,
       bias,
       reverse,
       quantizedLstmParams});
  inference_weight_desc.initialize_weight_src();
  inference_weight_desc.initialize_attribute();
  inference_weight_desc.set_expected_weights_desc();
  return inference_weight_desc.get_and_save_lstm_packed_weight();
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
    const bool train,
    const QuantizedLstmParams& quantizedLstmParams) {
  TORCH_CHECK(
      weight_ih.scalar_type() == weight_hh.scalar_type(),
      "Expected weight_ih and weight_hh to be the same scalar type");
  // TODO: This is a workaround. In training, weight prepack is not enabled
  // Will remove this when weight prepack is enabled.
  if (train) {
    CommonLstmWeightDesc train_weight_desc(
        {weight_ih,
         weight_hh,
         input_size,
         num_gates,
         hidden_size,
         output_sizes,
         src_layer,
         src_iter,
         src_iter_c,
         bias,
         reverse,
         quantizedLstmParams});
    train_weight_desc.initialize_weight_src();
    train_weight_desc.initialize_attribute();
    return train_weight_desc.get_lstm_public_weight();
  }

  auto dtype = weight_ih.scalar_type();
  switch (dtype) {
    case at::ScalarType::Float:
    case at::ScalarType::BFloat16:
      return LSTM_PACKED_WEIGHT(LstmDtype::Float);
    case at::ScalarType::QInt8:
    case at::ScalarType::QUInt8:
      return LSTM_PACKED_WEIGHT(LstmDtype::Quantized);
    default:
      TORCH_CHECK(false, "Invalid data type ", dtype);
  }
}

ideep::tensor::desc get_conv_transpose_expected_weights_desc(
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

/**
 * Synchronize FP32 master weight back to BF16 weight.
 * After updating FP32 master weights by grads,
 * FP32 master weights should be written back to BF16 weights
 * in the model.
 *
 *@param master_weight Master FP32 weight
 *@param bf16_weight BF16 weight
 */
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

/**
 * Synchronize FP32 master weight back to FP16 weight.
 * After updating FP32 master weights by grads,
 * FP32 master weights should be written back to FP16 weights
 * in the model.
 *
 *@param master_weight Master FP32 weight
 *@param fp16_weight FP16 weight
 */
void sync_master_weight_to_fp16(
    const at::Tensor master_weight,
    at::Tensor& fp16_weight) {
  TORCH_CHECK(
      master_weight.sizes() == fp16_weight.sizes(),
      "expected master weight has same sizes with fp16 weight");
  TORCH_CHECK(
      master_weight.scalar_type() == at::kFloat &&
          fp16_weight.scalar_type() == at::kHalf,
      "expected master weght has same dims with fp16 weight");
  auto w_master = itensor_view_from_dense(master_weight);
  auto w_fp16 = itensor_view_from_dense(fp16_weight);
  w_fp16.feed_from(w_master);
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

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "sync_master_weight_to_fp16(Tensor master_weight, Tensor fp16_weight) "
      "-> ()",
      torch_ipex::cpu::sync_master_weight_to_fp16);
}

} // namespace
