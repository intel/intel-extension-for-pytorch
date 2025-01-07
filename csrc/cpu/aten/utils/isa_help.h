#pragma once

#include <ATen/Tensor.h>
#include <Macros.h>
#include <dyndisp/DispatchStub.h>
#include <torch/all.h>
#include <string>

namespace torch_ipex {
namespace cpu {

IPEX_API std::string get_current_onednn_isa_level();

IPEX_API std::string get_current_isa_level();
IPEX_API std::string get_highest_cpu_support_isa_level();
IPEX_API std::string get_highest_binary_support_isa_level();

namespace {

IPEX_API std::string get_current_isa_level_kernel_impl();

}

using get_current_isa_level_kernel_fn = std::string (*)();
IPEX_DECLARE_DISPATCH(
    get_current_isa_level_kernel_fn,
    get_current_isa_level_kernel_stub);

} // namespace cpu
} // namespace torch_ipex