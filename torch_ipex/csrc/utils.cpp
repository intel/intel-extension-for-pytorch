#include "utils.h"

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

namespace torch_ipex {

thread_local IPEXFuncStatus g_current_ipex_func_stat = IPEXFuncStatus::IPEX_SUCCESS;

IPEXFuncStatus get_ipex_func_status() {
  return g_current_ipex_func_stat;
}

void set_ipex_func_status(IPEXFuncStatus ipex_fun_stat) {
  g_current_ipex_func_stat = ipex_fun_stat;
}

void reset_ipex_func_status() {
  set_ipex_func_status(IPEXFuncStatus::IPEX_SUCCESS);
}

bool is_ipex_func_success() {
  return g_current_ipex_func_stat == IPEXFuncStatus::IPEX_SUCCESS;
}

bool is_scalar_tensor(const at::Tensor& tensor) {
  auto strides = tensor.strides();
  for (int i = 0; i < strides.size(); i++) {
    if (strides[i] != 0) return false;
  }

  return tensor.numel() == 1;
}

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

bool is_transposed_2d(const at::Tensor& tensor){
  return (
    tensor.ndimension() == 2 &&
    tensor.stride(0) == 1 &&
    tensor.stride(1) == tensor.size(0)
  );
}

}  // namespace torch_ipex
