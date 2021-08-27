#pragma once

#include <ATen/core/grad_mode.h>
#include <utils/Macros.h>

struct IPEX_API InferenceMode {
  InferenceMode(bool enabled = true);

  ~InferenceMode() {
    _set_enabled(prev_mode);
  }

  static bool is_enabled();
  static void _set_enabled(bool enabled);

 private:
  bool prev_mode;
  at::AutoGradMode grad_mode;
};
