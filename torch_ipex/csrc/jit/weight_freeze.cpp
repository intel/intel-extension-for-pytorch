#include <torch/csrc/jit/ir.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "weight_freeze.h"

namespace torch { namespace jit {

//
// Just like quan-dequant process on param
// It should be put in utils, please :)
//
struct ParamValue {
  Value* definition;
  IValue slot;
};

//
// Recusively gather prim::GetAttr nodes and their correspond IValue
//
static void gatherParams(
    const script::Module& module,
    Value* module_value,
    std::vector<ParamValue>& params) {
  for (const Use& u : module_value->uses()) {
    if (u.user->kind() != prim::GetAttr) {
      continue;
    }
    const std::string& field = u.user->s(attr::name);
    if (const auto& sub = module.find_module(field)) {
      gatherParams(*sub, u.user->output(), params);
    } else if (auto slot = module.find_parameter(field)) {
      params.emplace_back(ParamValue{u.user->output(), slot});
    } else if (auto slot
        // XXX: attribute??
        = const_cast<script::Module&>(module).find_attribute(field)) {
      // For runnign_mean and running_var
      params.emplace_back(ParamValue{u.user->output(), slot});
    }
  }
}

//
// We don't know when to call this pass, so we need to check both symbols
// in pytorch or in mkldnn
//
bool nodeOfInterest(const Node* node) {
  return node->kind() == aten::conv2d
      || node->kind() == aten::batch_norm;
}

std::vector<ParamValue> getParamForFreeze(
    script::Method& method,
    const std::string& param_name) {
  std::vector<ParamValue> params;
  std::vector<ParamValue> ret;
  gatherParams(method.owner(), method.graph()->inputs().at(0), params);

  // We filter out params that should be freeze
  for (const auto& param : params) {
    if (!param.definition->type()->isSubtypeOf(TensorType::get()))
      continue;

    for(const auto& u : param.definition->uses()) {
      if (nodeOfInterest(u.user)
          && u.user->schema().arguments().at(u.offset).name() == param_name) {
        //
        // XXX: detach from param itself
        //
        ret.push_back({param.definition, param.slot.toTensor().detach()});
        break;
      }
    }
  }

  return ret;
}

std::vector<ParamValue> getFlagForFreeze(
    script::Method& method,
    const std::string& flag_name) {
  std::vector<ParamValue> params;
  std::vector<ParamValue> ret;
  gatherParams(method.owner(), method.graph()->inputs().at(0), params);

  // We filter out params that should be freeze
  for (const auto& param : params) {
    for(const auto& u : param.definition->uses()) {
      if (u.user->kind() == prim::If
          && u.user->input()->node()->s(attr::name) == flag_name) {
        // We don't need the value, because we will change it to arbitrary one.
        ret.push_back({param.definition, param.slot.toBool()});
        break;
      }
    }
  }

  return ret;
}

void insertConstantForParam(
    script::Method& method,
    const std::string& param_name) {
  const auto params = getParamForFreeze(method, param_name);
  auto g = method.graph();

  // Change all the accesses to constant
  for (const auto& param : params) {
    //
    // TODO: assert ref.defined()
    // to_mkldnn will disable infer a complete tensor from this point
    // we might not be able to fold BN at compiling time by constant
    // propagation
    //
    WithInsertPoint guard(param.definition->node()->next());
    auto n = tryInsertConstant(*g, param.slot);
    if (n) {
      param.definition->replaceAllUsesWith(*n);
      auto v = n.value();
      v->setDebugName(v->debugName() + '.' + param_name);
    }
  }
}

void FreezeParams(
    const script::Module& moduleObj,
    const std::string& method_name,
    const std::string& param_name
) {
  script::Method method = moduleObj.get_method(method_name);
  insertConstantForParam(method, param_name);

  // TODO: DCE? CP?
}

//
// TODO: implement partial redundency elimination and move the possible
// reorder closer to its use and optimize further.
//
// XXX: Freeze a flag is not a common behavior, it's a workaround
//
TORCH_API void FreezeFlags(
    const script::Module& moduleObj,
    const std::string& method_name,
    const std::string& flag_name,
    bool value) {
  script::Method method = moduleObj.get_method(method_name);
  const auto params = getFlagForFreeze(method, flag_name);
  auto g = method.graph();

  // Change all the accesses to constant
  for (const auto& param : params) {
    //
    // TODO: assert ref.defined()
    // to_mkldnn will disable infer a complete tensor from this point
    // we might not be able to fold BN at compiling time by constant
    // propagation
    //
    WithInsertPoint guard(param.definition->node()->next());
    auto n = tryInsertConstant(*g, IValue(value));
    if (n) {
      param.definition->replaceAllUsesWith(*n);
      auto v = n.value();
      v->setDebugName(v->debugName() + '.' + flag_name);
    }
  }
}
}} // namespace torch::jit
