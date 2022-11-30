#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/PackedParams.h>

#include <quantized/QUtils.h>
#include <quantized/Quantizer.h>

#include "CatImpl.h"
#include "ReQuantization.h"
#include "comm/ParamUtils.h"
#include "utils/CustomOperatorRegistration.h"

using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

Tensor q_cat_dequantize(
    const c10::List<Tensor>& tensors,
    int64_t dim,
    c10::optional<double> scale,
    c10::optional<int64_t> zero_point) {
  double scale_inter =
      scale.has_value() ? scale.value() : tensors.get(0).q_scale();
  int64_t zero_point_inter = zero_point.has_value()
      ? zero_point.value()
      : tensors.get(0).q_zero_point();
  auto out = at::empty(
      {0},
      tensors.get(0).options().dtype(at::kFloat),
      MemoryFormat::Contiguous);
  at::AtenIpexTypeXPU::cat_(ITensorListRef(tensors).materialize(), dim, out);
  return out;
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("q_cat_dequantize", q_cat_dequantize);
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

Tensor q_cat(
    const c10::List<Tensor>& tensors,
    int64_t dim,
    c10::optional<double> scale,
    c10::optional<int64_t> zero_point) {
  double scale_out =
      scale.has_value() ? scale.value() : tensors.get(0).q_scale();
  int64_t zero_point_out = zero_point.has_value()
      ? zero_point.value()
      : tensors.get(0).q_zero_point();

  bool isBlockfmt =
      std::any_of(tensors.begin(), tensors.end(), [](const Tensor& t) {
        return xpu::oneDNN::is_onednn_layout(t);
      });
  auto out = at::_empty_affine_quantized(
      {0},
      tensors.get(0).options().dtype(toQIntType(tensors.get(0).scalar_type())),
      scale_out,
      zero_point_out,
      MemoryFormat::Contiguous);

  std::vector<Tensor> tensors_;
  if (!isBlockfmt) {
    at::AtenIpexTypeXPU::cat_(ITensorListRef(tensors).materialize(), dim, out);
  } else {
    // This is a workaroud for oneDNN symmetric INT8, will remove it after
    // oneDNN Asymmetric INT8 is ready.
    zero_point_out = 0;

    out = at::_empty_affine_quantized(
        {0},
        tensors.get(0).options().dtype(
            toQIntType(tensors.get(0).scalar_type())),
        scale_out,
        zero_point_out,
        MemoryFormat::Contiguous);
    std::vector<Tensor> tensors_;
    for (int i = 0; i < tensors.size(); i++) {
      auto src = tensors.get(i);
      auto dst = requantize(src, scale_out, zero_point_out);
      tensors_.push_back(dst);
    }
    TensorList tensors_cat_array(tensors_);
    ITensorListRef tensors_ref = ITensorListRef(tensors_cat_array);
    xpu::oneDNN::concat(out, tensors_ref.materialize(), dim);
  }
  return out;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("quantized::cat", q_cat);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
