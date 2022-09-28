#pragma once

#include <torch/csrc/jit/tensorexpr/external_functions.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

struct NNCOperatorRegister {
  NNCOperatorRegister(
      const char* sig_literal,
      const char* external_func_name,
      torch::jit::tensorexpr::NNCLoweringFunction lowering_func,
      torch::jit::tensorexpr::NNCExternalFunction external_func);
  NNCOperatorRegister(
      const char* sig_literal,
      torch::jit::tensorexpr::NNCLoweringFunction lowering_func);

 private:
  void registerExternalCallOp(
      const char* sig_literal,
      const char* external_func_name,
      torch::jit::tensorexpr::NNCLoweringFunction lowering_func,
      torch::jit::tensorexpr::NNCExternalFunction external_func);
  void registerLoweringOp(
      const char* sig_literal,
      torch::jit::tensorexpr::NNCLoweringFunction lowering_func);
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
