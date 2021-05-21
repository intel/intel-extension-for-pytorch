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
    auto weight_ = IS_CONTIGUOUS_ANY(weight) ? weight : weight.contiguous();
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

}  // namespace cpu
}  // namespace torch_ipex
