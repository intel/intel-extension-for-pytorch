#include <torch/csrc/jit/api/module.h>

namespace torch_ipex {
namespace utils {

TORCH_API int onednn_set_verbose(int level);
TORCH_API bool onednn_has_bf16_type_support();
TORCH_API bool onednn_has_fp16_type_support();

} // namespace utils
} // namespace torch_ipex
