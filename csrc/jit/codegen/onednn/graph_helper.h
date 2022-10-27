#pragma once

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include "codegen/onednn/operator.h"

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

struct OpPartitionMap {
  void add(uint64_t opId, uint64_t partitionId) {
    opmap[opId] = partitionId;
  }
  void add(torch::jit::Node* n, uint64_t partitionId) {
    add(Operator::getId(n), partitionId);
  }
  bool has(uint64_t opId) {
    return opmap.count(opId) > 0;
  }
  bool has(torch::jit::Node* n) {
    return has(Operator::getId(n));
  }
  uint64_t get(uint64_t opId) {
    return opmap[opId];
  }
  uint64_t get(torch::jit::Node* n) {
    auto opId = Operator::getId(n);
    TORCH_CHECK(
        has(opId),
        "Node ",
        n->kind().toQualString(),
        " does not belong to any LLGA partition");
    return get(opId);
  }

 private:
  std::unordered_map<uint64_t, uint64_t> opmap;
};

class LlgaGraphHelper {
 public:
  LlgaGraphHelper(
      const std::shared_ptr<torch::jit::Graph>& graph,
      dnnl::graph::partition::policy policy =
          dnnl::graph::partition::policy::fusion);

  bool shouldMerge(torch::jit::Node* toMerge, torch::jit::Node* subgraph);

  bool shouldConsiderForMerge(torch::jit::Node* node);

  torch::jit::Node* createSingletonSubgraph(
      torch::jit::Node* n,
      torch::jit::AliasDb& db);

  void mergeNodeIntoSubgraph(
      torch::jit::Node* toMerge,
      torch::jit::Node* subgraphNode,
      torch::jit::AliasDb& db);

  void unmergeIfAnyNodeIsMissing(torch::jit::Node* subgraphNode);

  static bool isLlgaSubgraph(const torch::jit::Node* node);

  std::vector<dnnl::graph::partition> getPartitions() const;

  std::map<size_t, torch::jit::Value*> getTensorIdToValue() const;

  dnnl::graph::op createLlgaOp(torch::jit::Node* node);

  Operator createOperator(torch::jit::Node* node) const;

  bool isSupported(torch::jit::Node* node) const;

 private:
  size_t countSupportedOps(
      const std::shared_ptr<torch::jit::Graph>& graph) const;

  bool isSingleQuantDequantTo(torch::jit::Node* node);

  std::unique_ptr<torch::jit::AliasDb> aliasDb_ = nullptr;

  OpPartitionMap opToOwningPartition_;
  std::vector<dnnl::graph::partition> partitions_;
  std::map<size_t, torch::jit::Value*>
      tensorIdToValue_; // map from tensorId to torch::jit::Value
};

class LlgaNodeWrapper {
 public:
  LlgaNodeWrapper(const torch::jit::Node* node);

  void setOpaqueLayout(size_t offset);

  bool useOpaqueLayout(size_t offset) const;

  friend class LlgaGraphHelper;

 private:
  void initOutputLayouts();

  torch::jit::Node* n;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
