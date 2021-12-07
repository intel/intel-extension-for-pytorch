//[This file is from https://github.com/pytorch/pytorch/pull/63198/files and
// change it to adapt to CPU and IPEX]
#include "concat_linear.h"
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <unordered_set>
#include <vector>

#include "csrc/aten/cpu/WeightPack.h"

namespace torch {
namespace jit {
namespace {

using Tensor = at::Tensor;

class ConcatLinearLayers {
 public:
  explicit ConcatLinearLayers(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run() {
    handleBlockAndSubblocks(graph_->block());
    return graph_modified;
  }

  AliasDb* getAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  // torch/csrc/jit/passes/utils/optimization_utils.h is not in the current
  // torch, so add "nonConstantParameters" here ToDo will remove it when there
  // optimization_utils.h for torch
  bool nonConstantParameters(Node* n) {
    // Checks if the parameters, not including the
    // first param are all constants.
    for (size_t i = 1; i < n->inputs().size(); i++) {
      if (n->inputs().at(i)->node()->kind() != prim::Constant) {
        return true;
      }
    }
    return false;
  }

  void collectConstantLinearLayers(
      Block* b,
      std::unordered_map<Value*, std::vector<Node*>>& grouped_linear_layers,
      std::vector<Value*>& ordered_tensor_inputs) {
    // We are using an ordered list so that we only have to
    // check if moving items forward is a valid move, not
    // backwards. Otherwise we need to rebuild the aliasDb when we add values.

    for (Node* n : b->nodes()) {
      // Grouping together all linear layers that use the same Tensor for input
      if (n->kind() != aten::linear &&
          n->kind() != Symbol::fromQualString("torch_ipex::ipex_linear")) {
        continue;
      }

      auto weight = n->namedInput("weight");
      auto bias = n->namedInput("bias");
      if (weight->type() == NoneType::get() ||
          bias->type() == NoneType::get()) {
        continue;
      }

      if (nonConstantParameters(n)) {
        continue;
      }

      Value* linear_input = n->inputs().at(0);
      if (grouped_linear_layers.find(linear_input) ==
          grouped_linear_layers.cend()) {
        grouped_linear_layers.insert({linear_input, std::vector<Node*>()});
        ordered_tensor_inputs.push_back(linear_input);
      }
      grouped_linear_layers.find(linear_input)->second.push_back(n);
    }
  }

  void mergeLinearLayers(std::vector<Node*>& compatible_layers) {
    graph_modified = true;
    assert(!compatible_layers.empty());
    Node* base_node = compatible_layers[0];

    // Scope needed to make sure we free the WithInsertPoint guard
    // and reset the insert point before we delete `base_node`
    Node* linear_node = nullptr;
    {
      WithInsertPoint guard(base_node);
      std::vector<Tensor> weight_list;
      if (base_node->kind() == aten::linear) {
        weight_list = c10::fmap(compatible_layers, [](Node* n) {
          return constant_as<Tensor>(n->namedInput("weight")).value();
        });
      } else {
        // always unpack to contiguous tensor with no transposed, and the origin
        // weight dtype is same as packed weight.
        weight_list = c10::fmap(compatible_layers, [](Node* n) {
          return torch_ipex::cpu::linear_weight_unpack(
              constant_as<Tensor>(n->namedInput("weight")).value(),
              constant_as<int64_t>(n->inputs().at(2)).value(),
              constant_as<int64_t>(n->inputs().at(3)).value(),
              false,
              c10::nullopt);
        });
      }

      Tensor cat_weight = at::cat(weight_list, /*dim=*/0);

      auto bias_list = c10::fmap(compatible_layers, [](Node* n) {
        return constant_as<Tensor>(n->namedInput("bias")).value();
      });
      Tensor cat_bias = at::cat(bias_list, /*dim=*/0);
      Value* cat_bias_value = graph_->insertConstant(cat_bias);

      auto tensor_input = base_node->inputs().at(0);
      if (base_node->kind() == aten::linear) {
        Value* cat_weight_value = graph_->insertConstant(cat_weight);
        std::vector<Value*> linear_in = {
            tensor_input, cat_weight_value, cat_bias_value};
        linear_node = graph_->create(aten::linear, linear_in);
      } else {
        auto packed_cat_weight =
            torch_ipex::cpu::linear_weight_pack(cat_weight, c10::nullopt);
        Value* cat_weight_value = graph_->insertConstant(packed_cat_weight);
        Value* out_features_value = graph_->insertConstant(cat_weight.size(0));
        Value* in_features_value = graph_->insertConstant(cat_weight.size(1));
        std::vector<Value*> linear_in = {
            tensor_input,
            cat_weight_value,
            out_features_value,
            in_features_value,
            cat_bias_value};
        linear_node = graph_->create(
            Symbol::fromQualString("torch_ipex::ipex_linear"), linear_in);
      }
      auto input_size_option = base_node->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      // set output sizes
      if (input_size_option.has_value()) {
        auto input_size_value = input_size_option.value();
        input_size_value[input_size_value.size() - 1] = cat_weight.size(0);
        linear_node->output(0)->setType(
            base_node->output(0)->type()->expect<TensorType>()->withSizes(
                input_size_value));
      }
      linear_node->insertBefore(base_node);
    }

    // Update the outputs of the nodes
    WithInsertPoint guard2(linear_node);
    Value* neg1 = graph_->insertConstant(-1);
    Value* one = graph_->insertConstant(1);

    int64_t slice_start = 0;
    Value* slice_start_val = graph_->insertConstant(0);

    for (Node* orig_node : compatible_layers) {
      // for each node in the compatible_layers list,
      // slide the output of the combined linear layer
      // and use it instead of the output of the original node

      int64_t slice_end;
      if (orig_node->kind() == aten::linear) {
        Tensor weight_tensor =
            constant_as<Tensor>(orig_node->namedInput("weight")).value();
        slice_end = slice_start + weight_tensor.size(0);
      } else {
        slice_end = slice_start +
            constant_as<int64_t>(orig_node->inputs().at(2)).value();
      }

      Value* slice_end_val = graph_->insertConstant(slice_end);

      Node* slice = graph_->create(
          aten::slice,
          {linear_node->output(), neg1, slice_start_val, slice_end_val, one});
      slice->output(0)->setType(
          orig_node->output(0)->type()->expect<TensorType>());
      slice->insertAfter(linear_node);
      orig_node->replaceAllUsesWith(slice);
      orig_node->destroy();

      slice_start = slice_end;
      slice_start_val = slice_end_val;
    }
  }

  bool isNonZeroDimEqual(Tensor& tensor_a, Tensor& tensor_b) {
    if (tensor_a.dim() != tensor_b.dim()) {
      return false;
    }
    for (int64_t i = 1; i < tensor_a.dim(); i++) {
      if (tensor_a.size(i) != tensor_b.size(i)) {
        return false;
      }
    }
    return true;
  }

  // Check the linear_layer_group of a tensor to find ones that can be
  // combined
  void collectAndMergeLinearLayers(std::vector<Node*>& linear_layer_group) {
    std::unordered_set<Node*> checked_nodes;

    for (size_t i = 0; i < linear_layer_group.size(); i++) {
      Node* base_node = linear_layer_group[i];
      if (checked_nodes.count(base_node) != 0) {
        continue;
      }

      std::vector<Node*> compatible_layers;
      compatible_layers.push_back(base_node);

      auto base_weight =
          constant_as<Tensor>(base_node->namedInput("weight")).value();
      auto base_bias =
          constant_as<Tensor>(base_node->namedInput("bias")).value();

      // Now iterate over the rest of the users of the set to
      // see if there is anything that we can coaleasce `base_node` with.
      for (size_t j = i + 1; j < linear_layer_group.size(); j++) {
        auto node = linear_layer_group[j];
        // Only support all nodes ate aten::linear or torch_ipex::ipex_linear
        if (node->kind() != base_node->kind()) {
          continue;
        }
        if (checked_nodes.count(node) != 0) {
          continue;
        }
        auto weight = constant_as<Tensor>(node->namedInput("weight")).value();
        auto bias = constant_as<Tensor>(node->namedInput("bias")).value();

        // For now we will just keep it simple and require matching types
        // Type promotion might cause performance to actually decrease.
        if (base_weight.dtype() != weight.dtype() ||
            base_weight.device() != weight.device() ||
            base_bias.dtype() != bias.dtype() ||
            base_bias.device() != bias.device()) {
          continue;
        }

        if (node->kind() == aten::linear) {
          if (!isNonZeroDimEqual(base_weight, weight) ||
              !isNonZeroDimEqual(base_bias, bias)) {
            continue;
          }
        } else {
          // for torch_ipex::ipex_linear, we only need to check the in_features
          // are same size between two nodes and bias size.
          if (constant_as<int64_t>(base_node->inputs().at(3)).value() !=
                  constant_as<int64_t>(node->inputs().at(3)).value() ||
              !isNonZeroDimEqual(base_bias, bias)) {
            continue;
          }
        }
        bool can_move_before_all = true;
        for (auto n : compatible_layers) {
          can_move_before_all &=
              getAliasDb()->moveBeforeTopologicallyValid(node, n);
        }
        if (!can_move_before_all) {
          continue;
        }

        // Found a node that is eligible for combination
        compatible_layers.push_back(node);
        checked_nodes.insert(node);
      }
      if (compatible_layers.size() == 1) {
        continue; // No other layers to merge
      }
      mergeLinearLayers(compatible_layers);
    }
  }

  void handleBlockAndSubblocks(Block* block) {
    for (auto node : block->nodes()) {
      for (Block* subblock : node->blocks()) {
        handleBlockAndSubblocks(subblock);
      }
    }

    // Processing for the block itself
    std::unordered_map<Value*, std::vector<Node*>> grouped_linear_layers;
    std::vector<Value*> ordered_tensor_inputs;
    collectConstantLinearLayers(
        block, grouped_linear_layers, ordered_tensor_inputs);

    // Reverse topological ordering is used to prevent the need to
    // update the aliasDB
    for (auto tensor_it = ordered_tensor_inputs.rbegin();
         tensor_it != ordered_tensor_inputs.rend();
         ++tensor_it) {
      collectAndMergeLinearLayers(grouped_linear_layers.at(*tensor_it));
    }
  }

 private:
  std::shared_ptr<Graph> graph_;
  bool graph_modified = false;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
};
} // namespace

TORCH_API bool FrozenConcatLinear(std::shared_ptr<Graph>& graph) {
  ConcatLinearLayers concatLayers(graph);
  GRAPH_DUMP("Before FrozenConcatLinear", graph);
  bool changed = concatLayers.run();
  if (changed) {
    GRAPH_DUMP("After FrozenConcatLinear", graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch
