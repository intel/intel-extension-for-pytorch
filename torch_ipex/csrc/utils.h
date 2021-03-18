#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {

enum DPCPPSubDev {
  CPU,
};

enum IPEXFuncStatus {
  IPEX_SUCCESS,
  IPEX_UNIMPLEMENTED,
  IPEX_FALLBACK
};

IPEXFuncStatus get_ipex_func_status();
bool is_ipex_func_success();
void reset_ipex_func_status();
void set_ipex_func_status(IPEXFuncStatus ipex_fun_status);
const char* scalarTypeName(const at::ScalarType type);
const char* LayoutName(const at::Layout layout);

} // namespace torch_ipex
