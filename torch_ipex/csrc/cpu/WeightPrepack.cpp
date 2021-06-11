#include "WeightPrepack.h"
#include "mkldnn/MKLDNNCommon.h"
#include "torch_ipex/csrc/utils.h"

namespace torch_ipex {
namespace cpu {

namespace {

using weakref_type = c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using val_blocked = std::tuple<weakref_type, ideep::tensor>;
thread_local std::unordered_map<c10::TensorImpl *, val_blocked> cached_weights;

}  // namespace

ideep::tensor get_conv_prepacked_weight(
    const ideep::tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr) {
  auto it = cached_weights.find(weight.unsafeGetTensorImpl());
  if (it != cached_weights.end()) {
    return std::get<1>(it->second);
  } else { 
    auto weight_ = IS_CONTIGUOUS_ANY(weight) ? weight : weight.contiguous(weight.suggest_memory_format());
    ideep::tensor w = at::native::itensor_view_from_dense(weight_);
    // TODO: 3d check
    bool is_channels_last = input.get_desc().is_nhwc();
    ideep::tensor::desc packed_desc;
    if (is_channels_last) {
      packed_desc = ideep::convolution_forward::expected_weights_desc<true>(
        w.get_dims(),
        w.get_data_type(),
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward,
        input.get_data_type(),
        input.get_dims(),
        attr);
    } else {
      packed_desc = ideep::convolution_forward::expected_weights_desc<false>(
        w.get_dims(),
        w.get_data_type(),
        stride.vec(),
        padding.vec(),
        padding.vec(),
        dilation.vec(),
        groups,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward,
        input.get_data_type(),
        input.get_dims(),
        attr);
    }
    ideep::tensor result;
    result.init(packed_desc);
    result.feed_from(w);
    cached_weights.emplace(
        weight.unsafeGetTensorImpl(),
        val_blocked{weakref_type(weight.getIntrusivePtr()), result});
    return result;
  }
}

ideep::tensor get_linear_prepacked_weight(
    const ideep::tensor& input,
    const at::Tensor& weight) {
  auto it = cached_weights.find(weight.unsafeGetTensorImpl());
  if (it != cached_weights.end()) {
    return std::get<1>(it->second);
  } else {
    auto weight_ = weight.is_contiguous() ? weight : weight.contiguous();
    ideep::tensor w = at::native::itensor_view_from_dense(weight_);
    auto packed_desc = ideep::inner_product_forward::expected_weights_desc(
        w.get_dims(),
        input.get_dims(),
        w.get_data_type(),
        input.get_data_type()); 
    ideep::tensor result;
    result.init(packed_desc);
    result.feed_from(w);
    cached_weights.emplace(
        weight.unsafeGetTensorImpl(),
        val_blocked{weakref_type(weight.getIntrusivePtr()), result});
    return result;
  }
}

// Create mkldnn memory view from ATen tensor
inline ideep::tensor get_mkldnn_tensor_view(
    const at::Tensor& tensor, const ideep::tensor::desc& desc) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK(
      tensor.layout() == at::Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float || tensor.scalar_type() == at::ScalarType::BFloat16,
             "itensor_view_from_dense expects float or bfloat16 tensor input");
  //TORCH_INTERNAL_ASSERT(at::impl::variable_excluded_from_dispatch());
  if (tensor.scalar_type() == at::ScalarType::Float){
    return {desc, tensor.template data_ptr<float>()};
  } else {
    return {desc, tensor.template data_ptr<c10::BFloat16>()};
  }
}

bool is_prepacked(const at::Tensor& weight) {
  auto it = cached_weights.find(weight.unsafeGetTensorImpl());
  return it == cached_weights.end() ? false : true;
}

std::tuple<ideep::tensor, ideep::tensor> get_lstm_prepacked_weight(
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
    const bool reverse) {
  auto it_i = cached_weights.find(weight_ih.unsafeGetTensorImpl());
  auto it_h = cached_weights.find(weight_hh.unsafeGetTensorImpl());

  bool all_in_cache = it_i != cached_weights.end() && it_h != cached_weights.end();
  bool all_miss = it_i == cached_weights.end() && it_h == cached_weights.end();
  TORCH_CHECK(all_in_cache || all_miss, "both of the weights of LSTM should be cached or neither should be cached");

  if (it_i != cached_weights.end()) {
    ideep::tensor w_ih = std::get<1>(it_i->second);
    ideep::tensor w_hh = std::get<1>(it_h->second);
    return std::make_tuple(w_ih, w_hh);
  } else {
    auto w1 = get_mkldnn_tensor_view(weight_ih, {{1, 1, input_size, num_gates, hidden_size}, at::native::get_mkldnn_dtype(weight_ih.scalar_type()), ideep::format_tag::ldgoi});
    auto w2 = get_mkldnn_tensor_view(weight_hh, {{1, 1, hidden_size, num_gates, hidden_size}, at::native::get_mkldnn_dtype(weight_hh.scalar_type()), ideep::format_tag::ldgoi});
  
    ideep::tensor::desc packed_desc_ih, packed_desc_hh;
    std::tie(packed_desc_ih, packed_desc_hh) = ideep::lstm_forward::expected_weights_desc(
        output_sizes,
        src_layer,
        src_iter,
        src_iter_c,
        w1,
        w2,
        bias,
        reverse);
    
    // Don't prepack when the weight is of rnn_packed format
    // When the weight is of rnn_packed format, if the seq_lens of
    // the input changes, the format of weight also changes.
    // oneDNN does not support reorder from rnn_packed back to public format.
    // LSTM based on BRGEMM kernel (on AVX512 and newest ISAs) will use blocked
    // format for weight of LSTM, which won't change when the input seq_lens changes.
    if (packed_desc_ih.is_rnn_packed() || packed_desc_hh.is_rnn_packed()) {
      return std::make_tuple(w1, w2);
    }

    ideep::tensor packed_ih {packed_desc_ih}; 
    ideep::tensor packed_hh {packed_desc_hh};

    packed_ih.feed_from(w1);
    packed_hh.feed_from(w2);

    cached_weights.emplace(
        weight_ih.unsafeGetTensorImpl(),
        val_blocked{weakref_type(weight_ih.getIntrusivePtr()), packed_ih});
    
    cached_weights.emplace(
        weight_hh.unsafeGetTensorImpl(),
        val_blocked{weakref_type(weight_hh.getIntrusivePtr()), packed_hh});

    return std::make_tuple(packed_ih, packed_hh);
  }
}

}  // namespace cpu
}  // namespace torch_ipex
