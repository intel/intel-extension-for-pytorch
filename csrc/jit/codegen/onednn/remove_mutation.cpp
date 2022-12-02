#include "remove_mutation.h"
#include <torch/csrc/jit/passes/remove_mutation.h>

namespace torch_ipex {
namespace jit {

bool maybeAliveAfterNode(
    torch::jit::AliasDb* aliasdb,
    torch::jit::Node* node,
    torch::jit::Value* v,
    torch::jit::Value* x) {
  torch::jit::Node* next_node = node->next();
  while (next_node != nullptr &&
         next_node->kind() != torch::jit::prim::Return) {
    for (torch::jit::Value* i : next_node->inputs()) {
      if (aliasdb->mayContainAlias(i, v) && i != x) {
        return true;
      }
    }
    next_node = next_node->next();
  }
  return false;
}
namespace fuser {
namespace onednn {

using namespace torch::jit;

bool IPEXRemoveMutation::removeTensorMutation() {
  return removeTensorMutation(graph_->block());
}

// This function is copied from
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/remove_mutation.cpp
Node* IPEXRemoveMutation::createSpecialMappedOp(Node* n) {
  WithInsertPoint guard(n);
  auto inputs = n->inputs();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Node* new_node;
  if (n->matches(
          "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")) {
    auto dtype = graph_->insert(prim::dtype, {inputs.at(0)});
    new_node = graph_
                   ->insert(
                       aten::full_like,
                       {inputs.at(0), inputs.at(1)},
                       {NamedValue("dtype", dtype)})
                   ->node();
    new_node->copyMetadata(n);
    new_node->output()->setType(n->output()->type());
  } else if (n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)")) {
    new_node = graph_->insert(aten::zeros_like, {n->inputs().at(0)})->node();
  } else if (
      n->matches(
          "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")) {
    // TODO: we should have normal_like operator
    // normal(float mean, float std, int[] size, *, Generator? generator=None,
    // ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    auto size = graph_->insert(aten::size, {n->inputs().at(0)});
    auto dtype = graph_->insert(prim::dtype, {n->inputs().at(0)});
    auto layout = graph_->insert(prim::layout, {n->inputs().at(0)});
    auto device = graph_->insert(prim::device, {n->inputs().at(0)});
    auto pin_memory = graph_->insert(aten::is_pinned, {n->inputs().at(0)});
    auto generator = graph_->insertConstant(IValue());
    new_node = graph_->insertNode(graph_->create(
        aten::normal,
        {n->inputs().at(1),
         n->inputs().at(2),
         size,
         generator,
         dtype,
         layout,
         device,
         pin_memory}));
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
  new_node->copyMetadata(n);
  new_node->output()->setType(n->output()->type());
  return new_node;
}

bool IPEXRemoveMutation::removeTensorMutation(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    for (Block* sub_block : node->blocks()) {
      changed |= removeTensorMutation(sub_block);
    }

    MutationRemover mr(graph_);
    if (!mr.inplaceOpVariant(node)) {
      continue;
    }

    Value* mutated_value = node->inputs().at(0);
    Value* output = node->output();
    if (maybeAliveAfterNode(getAliasDb(), node, mutated_value, output)) {
      continue;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* new_node;
    if (mr.isSpecialMappedOp(node)) {
      new_node = createSpecialMappedOp(node);
    } else {
      auto schema_name = node->schema().name();
      auto new_schema = schema_name.substr(0, schema_name.size() - 1);
      new_node = graph_->create(Symbol::fromQualString(new_schema), 1);
      new_node->copyMetadata(node);
      new_node->insertBefore(node);
      for (Value* input : node->inputs()) {
        new_node->addInput(input);
      }
      new_node->output()->setType(node->output()->type());

      // weird case where there is an inplace op and an equivalent functional op
      // of the same symbol, but they have different schemas
      if (!new_node->maybeOperator()) {
        new_node->destroy();
        continue;
      }
    }

    changed = true;
    mutated_value->replaceAllUsesAfterNodeWith(node, new_node->output());
    node->output()->replaceAllUsesWith(new_node->output());

    // We rewrite something like:
    // x = torch.zeros()
    // x.add_(1)
    // x.add_(2)
    // to:
    // x = torch.zeros()
    // x0 = x.add(1)
    // x0.add_(2)
    // For the remainder of the function, x0 will have the
    // same aliasing relationships as the original x.
    // To avoid rebuilding the entire alias db, we can replace
    // the memory DAG element of x with x0.
    getAliasDb()->replaceWithNewValue(mutated_value, new_node->output());

    // it is an invariant that all mutable types have an element in the memory
    // DAG so we must regive x an alias db element. We have already verified
    // that the mutated value is a fresh alias with a single use.
    getAliasDb()->createValue(mutated_value);

    node->destroy();

    // Set aliasDb_ to nullptr to ensure the aliasDb_ is always aligned with the
    // latest graph_
    aliasDb_ = nullptr;
  }
  return changed;
}

bool IPEXRemoveTensorMutation(const std::shared_ptr<Graph>& graph) {
  bool changed = RemoveTensorMutation(graph);
  IPEXRemoveMutation irm(graph);
  changed |= irm.removeTensorMutation();
  return changed;
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
