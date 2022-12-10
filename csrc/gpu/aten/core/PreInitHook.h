#pragma once

namespace xpu {
namespace dpcpp {

// Here we use a hook mechanism to register the lazy_init function to
// pre_init_hook. It is necessary to split front-end and back-end libraries from
// code and logic. It can guarantee
//   1) we can call lazy_init in the back-end if necessary.
//   2) we can only link back-end library independent of front-back library.

using InitFnPtr = void (*)();

void do_pre_init_hook();

void set_pre_init_hook_fn(InitFnPtr fn);

struct PreInitHookRegister {
  explicit PreInitHookRegister(InitFnPtr fn) {
    set_pre_init_hook_fn(fn);
  }
};

#define IPEX_REGISTER_PRE_INIT_HOOK(fn)                         \
  namespace {                                                   \
  static xpu::dpcpp::PreInitHookRegister g_pre_init_hook_d(fn); \
  }

} // namespace dpcpp
} // namespace xpu
