#pragma once

namespace xpu {
namespace dpcpp {

using InitFnPtr = void (*)();

void do_lazy_init();

void set_lazy_init_fn(InitFnPtr fn);

struct LazyInitRegister {
  explicit LazyInitRegister(InitFnPtr fn) {
    set_lazy_init_fn(fn);
  }
};

#define IPEX_REGISTER_LAZY_INIT(fn)          \
  namespace {                                \
  static LazyInitRegister g_lazy_init_d(fn); \
  }

} // namespace dpcpp
} // namespace xpu
