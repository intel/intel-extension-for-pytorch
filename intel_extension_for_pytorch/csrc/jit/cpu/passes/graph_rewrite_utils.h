#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

inline auto accumu_use_check = [](const Node* add_node,
                                  const Value* accumu_value) {
  bool accumu_same_used = false;
  auto accumu_uses = accumu_value->uses();
  std::for_each(accumu_uses.begin(), accumu_uses.end(), [&](Use& u) {
    // if one user is the after nodes of add. we can't write accumu.
    if (u.user != add_node && !u.user->isBefore(add_node)) {
      accumu_same_used = true;
    }
  });
  return accumu_same_used;
};

// op     Y
//   \   /
//    add
// output = op_output + alpha*Y
inline auto fuse_add_filter_v1 =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      auto accumu = match.values_map.at(vmap.at("accumu"));
      auto add_node = match.values_map.at(vmap.at("res"))->node();
      bool accumu_same_used = accumu_use_check(add_node, accumu);
      // accumu is used by other ops or it is a constant,i
      if (accumu_same_used || accumu->node()->kind() == prim::Constant ||
          !accumu->type()->cast<TensorType>()) {
        return false;
      }
      // check inputs of add have same shapes.
      auto size1_option = add_node->inputs()
                              .at(0)
                              ->type()
                              ->cast<TensorType>()
                              ->sizes()
                              .concrete_sizes();
      auto size2_option = add_node->inputs()
                              .at(1)
                              ->type()
                              ->cast<TensorType>()
                              ->sizes()
                              .concrete_sizes();
      // if we can't get the shape info, we can't do inplace fusion.
      if (!size1_option.has_value() || !size2_option.has_value()) {
        return false;
      }
      auto size1_vec = size1_option.value();
      auto size2_vec = size2_option.value();
      if (size1_vec.empty() || size2_vec.empty() || size1_vec != size2_vec) {
        return false;
      }
      return true;
    };

//  Y    op
//   \   /
//    add
// output = Y + alpha*op_output
inline auto fuse_add_filter_v2 =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      auto accumu = match.values_map.at(vmap.at("accumu"));
      auto add_node = match.values_map.at(vmap.at("res"))->node();
      bool accumu_same_used = accumu_use_check(add_node, accumu);
      // accumu is used by other ops or it is a constant,
      // we can't write it inplace.

      if (accumu_same_used || accumu->node()->kind() == prim::Constant ||
          !accumu->type()->cast<TensorType>()) {
        return false;
      }
      // check inputs of add have same shapes.
      auto size1_option = add_node->inputs()
                              .at(0)
                              ->type()
                              ->cast<TensorType>()
                              ->sizes()
                              .concrete_sizes();
      auto size2_option = add_node->inputs()
                              .at(1)
                              ->type()
                              ->cast<TensorType>()
                              ->sizes()
                              .concrete_sizes();
      // if we can't get the shape info, we can't do inplace fusion.
      if (!size1_option.has_value() || !size2_option.has_value()) {
        return false;
      }
      auto size1_vec = size1_option.value();
      auto size2_vec = size2_option.value();
      if (size1_vec.empty() || size2_vec.empty() || size1_vec != size2_vec) {
        return false;
      }
      // alpha is optional
      if (vmap.find("alpha") != vmap.end()) {
        auto alpha = toIValue(match.values_map.at(vmap.at("alpha")));
        if (alpha.has_value() && alpha.value().isDouble()) {
          auto alpha_ = alpha.value().toDouble();
          if (alpha_ != 1.0) {
            return false;
          }
        }
      }
      return true;
    };

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
