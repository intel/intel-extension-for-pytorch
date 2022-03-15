#pragma once

#include <ATen/Tensor.h>
#include <torch/extension.h>
#include <string>
#include "intel_extension_for_pytorch/csrc/dyndisp/DispatchStub.h"

namespace torch_ipex {
namespace cpu {

std::string get_current_isa_level();
std::string get_highest_cpu_support_isa_level();
std::string get_highest_binary_support_isa_level();

#if defined(DYN_DISP_BUILD)
namespace {
#endif

std::string get_current_isa_level_kernel_impl();

#if defined(DYN_DISP_BUILD)
}
#endif

using get_current_isa_level_kernel_fn = std::string (*)();
DECLARE_DISPATCH(
    get_current_isa_level_kernel_fn,
    get_current_isa_level_kernel_stub);

} // namespace cpu
} // namespace torch_ipex