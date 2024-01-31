#include <Macros.h>
#include <torch/csrc/jit/api/module.h>

namespace torch_ipex {
namespace utils {

IPEX_API bool isa_has_amx_fp16_support();
IPEX_API bool isa_has_avx512_fp16_support();
IPEX_API bool isa_has_amx_support();
IPEX_API bool isa_has_avx512_bf16_support();
IPEX_API bool isa_has_avx512_vnni_support();
IPEX_API bool isa_has_avx512_support();
IPEX_API bool isa_has_avx2_vnni_support();
IPEX_API bool isa_has_avx2_support();

} // namespace utils
} // namespace torch_ipex
