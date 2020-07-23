#include "Linear.h"

#include "Common.h"
#include "cpu/ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace linear {

dil::tensor linear_impl(
    const dil::tensor& x,
    const dil::tensor& w,
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

} // namespace linear  
} // namespace dbl
} // namespace cpu
} // namespace torch_ipex
