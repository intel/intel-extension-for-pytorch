#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/module.h>

namespace torch { namespace jit {
//
// Freezing Params pass is for user to enable fixed weight during inference
// as the weight is known to be constant, MKL-DNN op could safely assume the
// transform result of it will never change.
// The pass need to run on module level instead as a passes since we need to
// know which parameters to materialize
// 
// Like insert quant-dequant node pass, we took the same procedure
//
TORCH_API void FreezeParams(
    const Module& moduleObj,
    const std::string& method_name,
    const std::string& param_name);

TORCH_API void FreezeFlags(
    const Module& moduleObj,
    const std::string& method_name,
    const std::string& flag_name,
    bool value);
}} // namespace torch::jit
