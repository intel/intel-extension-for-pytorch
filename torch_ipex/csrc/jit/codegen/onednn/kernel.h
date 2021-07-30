#pragma once

#include "jit/codegen/onednn/graph_helper.h"
#include "torch_ipex/csrc/LlgaTensorImpl.h"
#include "torch_ipex/csrc/rw_lock.h"
#include <unordered_map>

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using ArgSpec = at::LlgaTensorDesc;
using ArgSpecs = std::vector<ArgSpec>;
using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using TensorArgs = std::vector<at::Tensor>;

class LlgaKernel {
public:
  explicit LlgaKernel(const Node *fusionNode);

  void run(Stack &stack);

  const std::string &debugName() const { return debugName_; }

  const std::string &profileName() const { return profileName_; }

private:
  bool useOpaqueLayout(size_t offset) const;

  int64_t getOutputDtype(size_t offset) const;

  // Get the scale, zp and dtype from the node on the graph
  // and save them in the spec to re-use during runtime to
  // create qtensor for output of public format
  ArgSpec getQuantizedSpec(ArgSpec spec, size_t offset) const;

  ArgSpecs specializeInputSpecs(const TensorArgs &inputs) const;

  ArgSpecs specializeOutputSpecs(const dnnl::graph::partition &partition,
                                 const ArgSpecs &inputSpecs) const;

  dnnl::graph::compiled_partition
  compile(const dnnl::graph::partition &partition);

  std::tuple<RunArgs, RunArgs> prepareRunArgs(const TensorArgs &inputs,
                                              TensorArgs &outputs) const;

  static std::string genDebugName() {
    static size_t debugId = 0;
    return "LlgaPartition_" + std::to_string(debugId++);
  }

  std::string genProfileName() {
    std::vector<std::string> op_list;
    for (auto *node : graph_->block()->nodes()) {
      if (node->kind().is_aten()) {
        op_list.push_back(node->kind().toUnqualString());
      }
    }
    return c10::Join("+", op_list);
  }

  static dnnl::graph::logical_tensor toLogicalTensor(const ArgSpec &s) {
    return s.logical_tensor();
  }

  void lock_read() { rw_mutex_.lock_read(); }

  void lock_write() { rw_mutex_.lock_write(); }

  void unlock_read() { rw_mutex_.unlock_read(); }

  void unlock_write() { rw_mutex_.unlock_write(); }

  at::Device device_ = at::kCPU;
  const Node *fusionNode_;
  std::shared_ptr<Graph> graph_;
  int64_t nInputs_ = 0;
  int64_t nOutputs_ = 0;
  dnnl::graph::partition partition_;
  dnnl::graph::compiled_partition compilation_;
  ArgSpecs inputSpecs_;
  ArgSpecs outputSpecs_;
  std::unordered_map<size_t, size_t> inplacePairs_; // output id -> input offset
  std::string debugName_;
  std::string profileName_;
  torch_ipex::ReadWriteMutex rw_mutex_;
  bool is_initialized_ = false;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
