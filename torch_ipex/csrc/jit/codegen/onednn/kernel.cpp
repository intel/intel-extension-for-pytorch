#include "jit/codegen/onednn/graph_helper.h"
#include "jit/codegen/onednn/kernel.h"
#include "jit/codegen/onednn/runtime.h"
#include "jit/codegen/onednn/subgraph_dtype_setter.h"

#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using namespace dnnl::graph;

using data_type = dnnl::graph::logical_tensor::data_type;

data_type getPropagateDataType(int64_t num) {
  switch(num) {
    case OutputDtype::uint8:
      return data_type::u8;
    case OutputDtype::int8:
      return data_type::s8;
    case OutputDtype::fp32:
      return data_type::f32;
    case OutputDtype::undef:
      return data_type::undef;
    default:
      TORCH_CHECK(false, "Not support propagate output type num ", num);
  }
}

// Build static mapping from output id to input offset
// in accordance with available inplace options
std::unordered_map<size_t, size_t> getInplacePairs(
  dnnl::graph::compiled_partition compilation,
  const ArgSpecs& inputSpecs) {
  std::unordered_map<size_t, size_t> inplacePairs; // output id -> input offset
  for (auto&& option : compilation.get_inplace_ports()) {
    size_t inputId = option.first;
    size_t outputId = option.second;
    auto inputSpecIter =
        std::find_if(inputSpecs.begin(), inputSpecs.end(), [&](auto& spec) {
          return spec.tid() == inputId;
        });
    TORCH_CHECK(inputSpecIter != inputSpecs.end(), "In-place input not found");
    auto inputOffset = inputSpecIter - inputSpecs.begin();
    inplacePairs[outputId] = inputOffset;
  }
  return inplacePairs;
}

LlgaKernel::LlgaKernel(const Node* fusionNode)
    : fusionNode_(fusionNode),
      graph_(fusionNode->g(attr::Subgraph)),
      nInputs_(graph_->inputs().size()),
      nOutputs_(graph_->outputs().size()),
      debugName_(genDebugName()) {
  // TODO: This is a workaround to recreate the partitions here. 
  // The ideal way is to use the partition serialization API (not available from LLGA now) 
  // to carry a serialized string representation from graph rewrite and deserialize it here.
  auto partitions = LlgaGraphHelper(graph_).getPartitions();
  TORCH_CHECK(
      partitions.size() == 1,
      "LLGA subgraph should contain only one partition");
  partition_ = partitions[0];
  GRAPH_DEBUG("Initialized ", debugName(), "\n", graph_->toString());
}

bool LlgaKernel::useOpaqueLayout(size_t offset) const {
  return LlgaNodeWrapper(fusionNode_).useOpaqueLayout(offset);
}

int64_t LlgaKernel::getOutputDtype(size_t offset) const {
  return LlgaNodeWrapper(fusionNode_).getOutputDtypes(offset);
}

ArgSpecs LlgaKernel::specializeInputSpecs(const TensorArgs& inputs) const {
  ArgSpecs inputSpecs;
  inputSpecs.reserve(nInputs_);
  for (size_t i = 0; i < nInputs_; i++) {
    auto spec = ArgSpec(graph_->inputs()[i]).supplementTensorInfo(inputs[i]);
    inputSpecs.emplace_back(spec);
  }
  return inputSpecs;
}

ArgSpecs LlgaKernel::specializeOutputSpecs(
    const partition& partition,
    const ArgSpecs& inputSpecs) const {
  auto inputs = fmap(inputSpecs, toLogicalTensor);
  auto outputs = fmap(graph_->outputs(), toLogicalTensor);
  partition.infer_shape(inputs, outputs);

  ArgSpecs outputSpecs;
  outputSpecs.reserve(nOutputs_);
  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = ArgSpec(outputs[i]);

    int64_t output_dtype = getOutputDtype(i);

    logical_tensor::data_type propagate_dtype = getPropagateDataType(output_dtype);
    if (propagate_dtype != logical_tensor::data_type::undef) {
      spec = spec.dtype(propagate_dtype);
    } else {
      spec = spec.dtype(inputSpecs[0].dtype());
    }

    if (useOpaqueLayout(i))
      spec = spec.any();
    outputSpecs.emplace_back(spec);
  }
  return outputSpecs;
}

std::tuple<RunArgs, RunArgs> LlgaKernel::prepareRunArgs(
    const TensorArgs& inputs,
    TensorArgs& outputs,
    const ArgSpecs& inputSpecs,
    const ArgSpecs& outputSpecs,
    const std::unordered_map<size_t, size_t>& inplacePairs) const {
  RunArgs runInputs, runOutputs;
  for (size_t i = 0; i < nInputs_; i++) {
    auto spec = inputSpecs[i];
    runInputs.push_back({spec.logical_tensor(), inputs[i].data_ptr()});
  }

  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = outputSpecs[i];
    auto opt = c10::TensorOptions(spec.aten_scalar_type()).device(device_);

    auto outputId = spec.tid();
    auto iter = inplacePairs.find(outputId);
    if (iter != inplacePairs.end()) {
      // output reuses one of input tensors
      auto inputOffset = iter->second;
      auto inputTensor = inputs[inputOffset];
      outputs.push_back(inputTensor);
      runOutputs.push_back({spec.logical_tensor(), inputTensor.data_ptr()});
    } else if (spec.is_opaque() || spec.is_quantized()) {
      auto tensor = at::empty_llga(spec, opt);
      outputs.push_back(tensor);
      runOutputs.push_back(at::llga_from_aten_tensor(tensor));
    } else {
      auto tensor = at::empty(spec.sizes(), opt);
      outputs.push_back(tensor);
      runOutputs.push_back({spec.logical_tensor(), tensor.data_ptr()});
    }
  }

  return std::make_tuple(runInputs, runOutputs);
}

compiled_partition LlgaKernel::compile(
  const partition& partition, 
  const ArgSpecs& inputSpecs,
  const ArgSpecs& outputSpecs) {
  auto inputs = fmap(inputSpecs, toLogicalTensor);
  auto outputs = fmap(outputSpecs, toLogicalTensor);
  auto compilation = partition.compile(inputs, outputs, Engine::getEngine());

  return compilation;
}

void LlgaKernel::run(Stack& stack) {
  GRAPH_DEBUG("In ", debugName(), "\n");

  // Grab input values from stack
  auto stackInputs = last(stack, nInputs_);
  auto inputs = fmap(stackInputs, [&](const IValue& v) {
    TORCH_CHECK(
        v.isTensor(), "Stack values for LLGA partition must be Tensor type");
    return v.toTensor();
  });

  GRAPH_DEBUG("Specializing input logical tensors");
  auto inputSpecs = specializeInputSpecs(inputs);
  
  GRAPH_DEBUG("Inferring output logical tensors");
  auto outputSpecs = specializeOutputSpecs(partition_, inputSpecs);
  
  GRAPH_DEBUG("Compiling partition");
  auto compilation = compile(partition_, inputSpecs, outputSpecs);
  // Since layouts of opaque outputs would be known after compilation,
  // we need to query them out from compilation and update outputSpecs
  for (size_t i = 0; i < nOutputs_; i++) {
    auto tid = outputSpecs[i].tid();
    outputSpecs[i] = compilation.query_logical_tensor(tid);
  }
  // output id -> input offset
  auto inplacePairs = getInplacePairs(compilation, inputSpecs); // output id -> input offset

  GRAPH_DEBUG("Preparing runtime tensors");
  TensorArgs outputs;
  RunArgs runInputs, runOutputs;
  std::tie(runInputs, runOutputs) = prepareRunArgs(inputs, outputs, inputSpecs, outputSpecs, inplacePairs);

  GRAPH_DEBUG("Executing partition");
  compilation.execute(Stream::getStream(), runInputs, runOutputs);
  GRAPH_DEBUG("Partition executed");

  // Update the stack.
  drop(stack, nInputs_);
  for (auto& o : outputs)
    push_one(stack, std::move(o));
  GRAPH_DEBUG("Stack updated");
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch