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

// A light-weight TORCH_CHECK that does not collect any backtrace info
#if defined(_DEBUG)
#define IPEX_CHECK(cond, ...)                                                  \
  if (!(cond)) {                                                               \
    throw std::runtime_error(                                                  \
      c10::detail::if_empty_then(                                              \
        c10::str(__VA_ARGS__),                                                 \
        "Expected " #cond " to be true, but got false."));                     \
  }
#else
// quick path of IPEX_CHECK without reporting message
#define IPEX_CHECK(cond, ...)                                                  \
  if (!(cond)) { throw std::exception(); }
#endif

} // namespace torch_ipex
