#include "utils.h"

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

namespace torch_ipex {

#define AT_FOR_AUTOCAST_SCALAR_TYPES(_) \
  _(int8, Char)                         \
  _(float32, Float)                     \
  _(bfloat16, BFloat16)

const char* scalarTypeName(const at::ScalarType type) {
  switch (type) {
#define DEFINE_CASE(ctype, name) \
  case at::ScalarType::name:     \
    return #ctype;
    AT_FOR_AUTOCAST_SCALAR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      throw std::runtime_error("unknown scalar type");
  }
}

const char* LayoutName(const at::Layout layout) {
  switch (layout) {
    case at::kStrided:
      return "torch.strided";
    case at::kMkldnn:
      return "torch._mkldnn";
    default:
      AT_ERROR("Unknown layout");
  }
}

bool is_transposed_2d(const at::Tensor& tensor) {
  return (
      tensor.ndimension() == 2 && tensor.stride(0) == 1 &&
      tensor.stride(1) == tensor.size(0));
}

} // namespace torch_ipex
