#pragma once

namespace xpu {
namespace dpcpp {

using FnPtr = void (*)();
void setLazyInit(FnPtr fn);

struct LazyInitRegister {
  explicit LazyInitRegister(FnPtr fn) {
    setLazyInit(fn);
  }
};

#define REGISTER_LAZY_INIT(fn)               \
  namespace {                                \
  static LazyInitRegister g_lazy_init_d(fn); \
  }

// Don't call back fn when fn is nullptr. It makes sure backend library
// libintel-ext-pt-gpu.so can be used independently.
#define LAZY_INIT_CALLBACK(fn) \
  if (fn) {                    \
    (*fn)();                   \
  }

} // namespace dpcpp
} // namespace xpu
