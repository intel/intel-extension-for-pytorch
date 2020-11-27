#include "Linear.h"

#include "Common.h"
#include "cpu/ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace linear {

dil::tensor linear_impl(
    const dil::tensor& x,
    dil::tensor& w,
    const c10::optional<dil::tensor>& b,
    const dil::scale_t& dst_scales,
    const dil::attr_t& attr) {

  dil::lowp_kind alowp_kind = dil::u8s8;
  if (dil::data_type::s8 == x.get_data_type()) {
    alowp_kind = dil::s8s8;
  }

  dil::tensor y;
  if (b.has_value()) {
    dil::inner_product_forward::compute(
        x,
        w,
        b.value(),
        y,
        dil::scale_t(),
        dil::scale_t(),
        dst_scales,
        attr,
        dil::prop_kind::forward,
        alowp_kind);
  } else {
    dil::inner_product_forward::compute(
        x,
        w,
        y,
        dil::scale_t(),
        dil::scale_t(),
        dst_scales,
        attr,
        dil::prop_kind::forward,
        alowp_kind);
  }
  return y;
}

void prepack_linear_weights(
    const at::Tensor& input,
    const dil::tensor& dil_input,
    const at::Tensor& weight) {
  if (!cpu::ShadeDataContext::isPackedTensor(weight)) {
    auto dil_weight = dbl::comm::try_gen_dil_tensor(weight);
    auto packed_desc = dil::inner_product_forward::expected_weights_desc(
      weight.sizes().vec(),
      input.sizes().vec(),
      dil_weight.get_data_type(),
      dil_input.get_data_type());

    dil::tensor packed_weight {packed_desc};
    
    if (dil_weight.has_scale()) {
      packed_weight.set_scale(dil_weight.get_scale());
    }
    packed_weight.feed_from(dil_weight);
    dbl::comm::equip_dil_buffer(weight, packed_weight);
    cpu::ShadeDataContext::setPackedTensor(weight, true);
  }
}

} // namespace linear  
} // namespace dbl
} // namespace cpu
} // namespace torch_ipex
