#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>

#include <oneDNN/oneDNN.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>

#include <quantized/QUtil.h>
#include "Cat.h"
#include "ReQuantization.h"
#include "comm/ParamUtils.h"

using namespace dnnl;
using namespace at::native;

namespace at {

using namespace AtenIpexTypeXPU::impl;

namespace AtenIpexTypeQuantizedXPU {

Tensor q_cat(
    const c10::List<Tensor>& tensors,
    int64_t dim,
    c10::optional<double> scale,
    c10::optional<int64_t> zero_point) {
  double scale_out =
      scale.has_value() ? scale.value() : tensors.get(0).q_scale();

  // This is a workaroud for oneDNN symmetric INT8, will remove it after oneDNN
  // Asymmetric INT8 is ready.
  int64_t zero_point_out = 0;

  std::vector<Tensor> tensors_;
  for (int i = 0; i < tensors.size(); i++) {
    auto src = tensors.get(i);
    auto dst = requantize(src, scale_out, zero_point_out);
    tensors_.push_back(dst);
  }
  TensorList tensors_cat_array(tensors_);
  auto out = _empty_affine_quantized(
      {0},
      tensors.get(0).options().dtype(toQIntType(tensors.get(0).scalar_type())),
      scale_out,
      zero_point_out,
      MemoryFormat::Contiguous);
  dnnl_cat(out, tensors_cat_array, tensors.size(), dim);
  return out;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("quantized::cat", q_cat);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
