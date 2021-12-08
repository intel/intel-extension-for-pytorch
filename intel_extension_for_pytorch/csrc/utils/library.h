#include <torch/library.h>

#include <c10/util/Logging.h>

// The flag is used to control the pytorch log level and defined at c10
extern int FLAGS_caffe2_log_level;

// This FLAGS_caffe2_log_level flag is used to control the log level and defined
// at Pytorch c10 library. The default is log level is warn. But it triggers the
// warn to override the kernel under a particular dispatch key. Unfortunately,
// we have to do this to capture aten operators. Hence, the behavior will
// trigger too many warning messages. It is terrible. So we have to temporarily
// promote the log level to error to avoid triggering warning message when we
// override the kernels. And after that, we need to restore the flag to the old
// value. Currently, we use the magic number 2 to represent the error log level
// because Pytorch does not expose a global variable to represent a particular
// log level.
#define _IPEX_TORCH_LIBRARY_IMPL(ns, k, m, uid)                                \
  static void C10_CONCATENATE(                                                 \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&);            \
  static void C10_CONCATENATE(                                                 \
      IPEX_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & lib) { \
    int org_FLAGS_caffe2_log_level = FLAGS_caffe2_log_level;                   \
    FLAGS_caffe2_log_level = 2;                                                \
    C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(lib);          \
    FLAGS_caffe2_log_level = org_FLAGS_caffe2_log_level;                       \
  }                                                                            \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(                \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(                      \
      torch::Library::IMPL,                                                    \
      c10::guts::if_constexpr<c10::impl::dispatch_key_allowlist_check(         \
          c10::DispatchKey::k)>(                                               \
          []() {                                                               \
            return &C10_CONCATENATE(                                           \
                IPEX_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid);              \
          },                                                                   \
          []() { return [](torch::Library&) -> void {}; }),                    \
      #ns,                                                                     \
      c10::make_optional(c10::DispatchKey::k),                                 \
      __FILE__,                                                                \
      __LINE__);                                                               \
  void C10_CONCATENATE(                                                        \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)

#define IPEX_TORCH_LIBRARY_IMPL(ns, k, m) \
  _IPEX_TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)
