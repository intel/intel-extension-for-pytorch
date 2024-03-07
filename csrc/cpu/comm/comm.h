#pragma once

namespace torch_ipex {
namespace cpu {
void barrier();
int get_world_size();
int get_rank();
} // namespace cpu
} // namespace torch_ipex