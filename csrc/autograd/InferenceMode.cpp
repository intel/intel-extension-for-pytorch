#include <c10/util/Exception.h>

#include <mutex>
#include "InferenceMode.h"

thread_local bool InferenceMode_enabled = false;
static std::once_flag announce;

InferenceMode::InferenceMode(bool enabled)
    : prev_mode(InferenceMode::is_enabled()),
      grad_mode(at::AutoGradMode(!enabled)) {
  std::call_once(announce, []() {
    TORCH_WARN(
        "`ipex.inference_mode` will be deprecated,",
        " to use `torch.inference_mode` after PyTorch 1.9");
  });
  _set_enabled(enabled);
}

bool InferenceMode::is_enabled() {
  return InferenceMode_enabled;
}

void InferenceMode::_set_enabled(bool enabled) {
  InferenceMode_enabled = enabled;
}
