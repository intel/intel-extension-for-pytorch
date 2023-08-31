#pragma once

#include <unordered_map>
#include <vector>
#include "codegen/LlgaTensorImpl.h"
#include "graph_helper.h"
#include "utils/rw_lock.h"

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace std {
template <>
struct hash<std::vector<int64_t>> {
  size_t operator()(const std::vector<int64_t>& key) const {
    size_t total = key.size();
    size_t sum = 0;
    if (total < 64) {
      for (size_t i = 0; i < total; i++) {
        sum += key[i] << i;
      }
    } else {
      size_t batch = total / 64;
      size_t remain = total % 64;
      for (size_t bs = 0; bs < batch; bs++) {
        for (size_t i = 0; i < 64; i++) {
          sum += key[bs * 64 + i] << i;
        }
      }
      for (size_t i = 0; i < remain; i++) {
        sum += key[batch * 64 + i] << i;
      }
    }
    return sum;
  }
};

} // namespace std

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using ArgSpec = LlgaTensorDesc;
using ArgSpecs = std::vector<ArgSpec>;
using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using TensorArgs = std::vector<at::Tensor>;

class LlgaKernel {
 public:
  explicit LlgaKernel(const torch::jit::Node* fusionNode);

  void run(torch::jit::Stack& stack);

  const std::string& debugName() const {
    return debugName_;
  }

  const std::string& profileName() const {
    return profileName_;
  }

 private:
  bool useOpaqueLayout(size_t offset) const;

  int64_t getOutputDtype(size_t offset) const;

  enum TypeOfOutputTensor {
    undefined,
    unwrappedInplaceCompute,
    quantizedInplaceCompute,
    unquantizedInplaceCompute,
    betweenPartitions,
    quantizedInputToFW,
    unquantizedInputToFW
  };

  struct cp_entry {
    dnnl::graph::compiled_partition cp_;
    RunArgs inputLLGATensors_;
    RunArgs outputLLGATensors_;
    ArgSpecs outputSpecs_;
  };

  // Get the scale, zp and dtype from the node on the graph
  // and save them in the spec to re-use during runtime to
  // create qtensor for output of public format
  ArgSpec getQuantizedSpec(ArgSpec spec, size_t offset) const;

  std::map<size_t, int64_t> initializeTensorIdToOccurence() const;

  // PyTorch copy constants inside the subgraph instead of referencing them.
  // Constants inputs to the partition are no longer in the graph->inputs().
  // Need use the tid retrieved from the partition to find the missing
  // constant inputs.

  ArgSpecs initializeInputSpecs(const TensorArgs& inputs);

  ArgSpecs initializeOutputSpecs(
      const TensorArgs& inputs,
      bool convertDimsToUnknown);

  std::pair<dnnl::graph::compiled_partition, ArgSpecs> compile(
      const dnnl::graph::partition& partition,
      const TensorArgs& inputs,
      ArgSpecs& inputSpecs);

  cp_entry& compileAndCache(torch::jit::Stack& stack, TensorArgs& outputs);

  void prepareRunArgs(
      RunArgs& inputLlgaTensors,
      RunArgs& outputLlgaTensors,
      const TensorArgs& inputs,
      TensorArgs& outputs,
      ArgSpecs& outputSpecs);

  void prepareAndCacheRunArgs(
      RunArgs& inputLlgaTensors,
      RunArgs& outputLlgaTensors,
      const TensorArgs& inputs,
      TensorArgs& outputs,
      ArgSpecs& inputSpecs,
      ArgSpecs& outputSpecs);

  static std::string genDebugName() {
    static size_t debugId = 0;
    return "LlgaPartition_" + std::to_string(debugId++);
  }

  bool inputValueIsNotUsedLater(size_t offset) const;

  std::string genProfileName() {
    std::vector<std::string> op_list;
    for (auto* node : graph_->block()->nodes()) {
      if (node->kind().is_aten()) {
        op_list.push_back(node->kind().toUnqualString());
      }
    }
    return c10::Join("+", op_list);
  }

  static dnnl::graph::logical_tensor toLogicalTensor(const ArgSpec& s) {
    return s.logical_tensor();
  }

  at::Device device_ = at::kCPU;
  const torch::jit::Node* fusionNode_;
  std::shared_ptr<torch::jit::Graph> graph_;
  int64_t nGraphInputs_ = 0; // number of inputs to graph_ on the IR
  int64_t nOutputs_ = 0;

  std::map<size_t, torch::jit::Value*> tensorIdToValue_;
  std::vector<int64_t> runArgsIdx_;
  dnnl::graph::partition partition_;
  // nPartitionInputs_ is the actual number of inputs to partition_ of graph_
  // needed by the backend.
  // nPartitionInputs_ = nGraphInputs_ + constantInputs_.size() since Constant
  // inputs are copied to the inside of the subgraph
  int64_t nPartitionInputs_;
  std::set<size_t> initializedInputIds_;
  std::vector<torch::jit::Value*> constantValues_;
  TensorArgs constantInputs_;

  // We'll do LRU without helper functions to minimize calls to the hash
  // function. Adopted from
  // https://github.com/lamerman/cpp-lru-cache/blob/master/include/lrucache.hpp
  // LRU cache is per-thread, so as to enable weight sharing among groups of
  // threads.
  using key_value_pair_t = std::pair<std::vector<int64_t>, cp_entry>;
  using list_iterator_t = std::list<key_value_pair_t>::iterator;
  static thread_local std::list<key_value_pair_t> cache_items_list_;
  static thread_local std::unordered_map<std::vector<int64_t>, list_iterator_t>
      cache_items_map_;
  static thread_local int capacity_;
  std::vector<std::vector<int64_t>> tracedInputShapes_;
  std::vector<std::vector<int64_t>> tracedInputStrides_;
  std::string debugName_;
  std::string profileName_;
  std::vector<TypeOfOutputTensor> outputTensorTypes_;
  std::once_flag constantSpecInitializedFlag_;
  std::once_flag tracedInputShapesInitialized_;
  std::vector<short> inplacePairOffsets_;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
