#pragma once
#include <Macros.h>

namespace torch_ipex {
namespace cpu {
IPEX_API void barrier();
IPEX_API int get_world_size();
IPEX_API int get_rank();
} // namespace cpu
} // namespace torch_ipex