#include "nnc_lowering_register.h"

#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/external_functions.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include "external_call/matmul_div.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

NNCOperatorRegister::NNCOperatorRegister(
    const char* sig_literal,
    const char* external_func_name,
    torch::jit::tensorexpr::NNCLoweringFunction lowering_func,
    torch::jit::tensorexpr::NNCExternalFunction external_func) {
  registerExternalCallOp(
      sig_literal, external_func_name, lowering_func, external_func);
}

NNCOperatorRegister::NNCOperatorRegister(
    const char* sig_literal,
    torch::jit::tensorexpr::NNCLoweringFunction lowering_func) {
  registerLoweringOp(sig_literal, lowering_func);
}

void NNCOperatorRegister::registerExternalCallOp(
    const char* sig_literal,
    const char* external_func_name,
    torch::jit::tensorexpr::NNCLoweringFunction lowering_func,
    torch::jit::tensorexpr::NNCExternalFunction external_func) {
  registerLoweringOp(sig_literal, lowering_func);

  auto& te_nnc_func_registry = torch::jit::tensorexpr::getNNCFunctionRegistry();
  te_nnc_func_registry[external_func_name] = external_func;
}

void NNCOperatorRegister::registerLoweringOp(
    const char* sig_literal,
    torch::jit::tensorexpr::NNCLoweringFunction lowering_func) {
  auto& te_lowering_registry = torch::jit::tensorexpr::getNNCLoweringRegistry();
  te_lowering_registry.insert(
      torch::jit::parseSchema(sig_literal), lowering_func);
}

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
