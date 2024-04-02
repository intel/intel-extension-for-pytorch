#include <Macros.h>
#include <torch/csrc/jit/api/module.h>

namespace torch_ipex {
namespace utils {

IPEX_API int onednn_set_verbose(int level);
IPEX_API bool onednn_has_bf16_type_support();
IPEX_API bool onednn_has_fp16_type_support();
IPEX_API bool onednn_has_fp8_type_support();

} // namespace utils
} // namespace torch_ipex
