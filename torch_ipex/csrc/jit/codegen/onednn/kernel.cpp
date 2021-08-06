#include "jit/codegen/onednn/kernel.h"
#include "jit/codegen/onednn/graph_helper.h"
#include "jit/codegen/onednn/operator.h"
#include "jit/codegen/onednn/runtime.h"
#include "jit/codegen/onednn/subgraph_dtype_setter.h"

#include <ATen/core/functional.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using namespace dnnl::graph;

using data_type = dnnl::graph::logical_tensor::data_type;

data_type getPropagateDataType(int64_t num) {
  switch (num) {
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
LlgaKernel::LlgaKernel(const Node *fusionNode)
    : fusionNode_(fusionNode), graph_(fusionNode->g(attr::Subgraph)),
      nInputs_(graph_->inputs().size()), nOutputs_(graph_->outputs().size()),
      debugName_(genDebugName()), profileName_(genProfileName()) {
  // TODO: This is a workaround to recreate the partitions here.
  // The ideal way is to use the partition serialization API (not available from
  // LLGA now) to carry a serialized string representation from graph rewrite
  // and deserialize it here.
  auto partitions = LlgaGraphHelper(graph_).getPartitions();
  TORCH_CHECK(partitions.size() == 1,
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

ArgSpec LlgaKernel::getQuantizedSpec(ArgSpec spec, size_t offset) const {
  auto node = graph_->outputs()[offset]->node();
  TORCH_CHECK(node->kind() == Symbol::aten("quantize_per_tensor") ||
                  node->kind() == Symbol::aten("quantize_per_channel"),
              "Int8 tensor must be the output from a quantize operator");

  if (node->kind() == Symbol::aten("quantize_per_tensor")) {
    spec = spec.set_quantizer(at::make_per_tensor_affine_quantizer(
        Operator::Float(node, /* offset */ 1),
        Operator::Int(node, /* offset */ 2),
        static_cast<at::ScalarType>(Operator::Int(node, /* offset */ 3))));
  } else {
    spec = spec.set_quantizer(at::make_per_channel_affine_quantizer(
        Operator::Tensor(node, /* offset */ 1),
        Operator::Tensor(node, /* offset */ 2),
        Operator::Int(node, /* offset */ 3),
        static_cast<at::ScalarType>(Operator::Int(node, /* offset */ 4))));
  }
  return spec;
}

ArgSpecs LlgaKernel::specializeInputSpecs(const TensorArgs &inputs) const {
  ArgSpecs inputSpecs;
  inputSpecs.reserve(nInputs_);
  for (size_t i = 0; i < nInputs_; i++) {
    auto spec = ArgSpec(graph_->inputs()[i]).supplementTensorInfo(inputs[i]);
    inputSpecs.emplace_back(spec);
  }
  return inputSpecs;
}

ArgSpecs LlgaKernel::specializeOutputSpecs(const partition &partition,
                                           const ArgSpecs &inputSpecs) const {
  auto inputs = fmap(inputSpecs, toLogicalTensor);
  auto outputs = fmap(graph_->outputs(), toLogicalTensor);
  partition.infer_shape(inputs, outputs);

  ArgSpecs outputSpecs;
  outputSpecs.reserve(nOutputs_);
  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = ArgSpec(outputs[i]);

    int64_t output_dtype = getOutputDtype(i);

    logical_tensor::data_type propagate_dtype =
        getPropagateDataType(output_dtype);
    if (propagate_dtype != logical_tensor::data_type::undef) {
      spec = spec.dtype(propagate_dtype);

      if (propagate_dtype == data_type::u8 ||
          propagate_dtype == data_type::s8) {
        spec = getQuantizedSpec(spec, i);
      }
    } else {
      // TODO: need run with profiling graph executor turned on to know the
      // exact output dtype. Hard-coded to f32 when can't infer from the node
      // that produces the output node. This will make bf16 fail to run.
      spec = spec.dtype(data_type::f32);
    }

    if (useOpaqueLayout(i))
      spec = spec.any();
    outputSpecs.emplace_back(spec);
  }
  return outputSpecs;
}

std::tuple<RunArgs, RunArgs>
LlgaKernel::prepareRunArgs(const TensorArgs &inputs,
                           TensorArgs &outputs) const {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("LLGA_bridge::prepareRunArgs", std::vector<c10::IValue>({}));
#endif
  RunArgs runInputs, runOutputs;
  for (size_t i = 0; i < nInputs_; i++) {
    auto spec = inputSpecs_[i];
    runInputs.push_back({spec.logical_tensor(), inputs[i].data_ptr()});
  }

  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = outputSpecs_[i];
    auto opt = c10::TensorOptions(spec.aten_scalar_type()).device(device_);

    auto outputId = spec.tid();
    auto iter = inplacePairs_.find(outputId);
    if (iter != inplacePairs_.end()) {
      // output reuses one of input tensors
      auto inputOffset = iter->second;
      auto inputTensor = inputs[inputOffset];
      outputs.push_back(inputTensor);
      runOutputs.push_back({spec.logical_tensor(), inputTensor.data_ptr()});
    } else if (spec.is_opaque()) {
      auto tensor = at::empty_llga(spec, opt);
      outputs.push_back(tensor);
      runOutputs.push_back(at::llga_from_aten_tensor(tensor));
    } else {
      if (spec.is_quantized()) {
        at::QuantizerPtr quantizer = spec.get_quantizer();
        auto qtensor = at::new_qtensor(spec.sizes(), opt, quantizer);
        outputs.push_back(qtensor);
        runOutputs.push_back({spec.logical_tensor(), qtensor.data_ptr()});
      } else {
        auto tensor = at::empty(spec.sizes(), opt);
        outputs.push_back(tensor);
        runOutputs.push_back({spec.logical_tensor(), tensor.data_ptr()});
      }
    }
  }

  return std::make_tuple(runInputs, runOutputs);
}

compiled_partition LlgaKernel::compile(const partition &partition) {
  auto inputs = fmap(inputSpecs_, toLogicalTensor);
  auto outputs = fmap(outputSpecs_, toLogicalTensor);
  auto compilation = partition.compile(inputs, outputs, Engine::getEngine());

  // Since layouts of opaque outputs would be known after compilation,
  // we need to query them out from compilation and update outputSpecs
  for (size_t i = 0; i < nOutputs_; i++) {
    auto tid = outputSpecs_[i].tid();
    outputSpecs_[i] =
        outputSpecs_[i].update_desc(compilation.query_logical_tensor(tid));
  }

  // Build static mapping from output id to input offset
  // in accordance with available inplace options
  for (auto &&option : compilation.get_inplace_ports()) {
    size_t inputId = option.first;
    size_t outputId = option.second;
    auto inputSpecIter =
        std::find_if(inputSpecs_.begin(), inputSpecs_.end(),
                     [&](auto &spec) { return spec.tid() == inputId; });
    TORCH_CHECK(inputSpecIter != inputSpecs_.end(), "In-place input not found");
    auto inputOffset = inputSpecIter - inputSpecs_.begin();
    inplacePairs_[outputId] = inputOffset;
  }

  return compilation;
}

void LlgaKernel::run(Stack &stack) {
  GRAPH_DEBUG("In ", debugName(), "\n");

  // Grab input values from stack
  auto stackInputs = last(stack, nInputs_);
  auto inputs = fmap(stackInputs, [&](const IValue &v) {
    TORCH_CHECK(v.isTensor(),
                "Stack values for LLGA partition must be Tensor type");
    return v.toTensor();
  });

  lock_read();
  if (is_initialized_) {
    unlock_read();
  } else {
    unlock_read();

    lock_write();
    if (!is_initialized_) {
      GRAPH_DEBUG("Specializing input logical tensors");
      inputSpecs_ = specializeInputSpecs(inputs);
      GRAPH_DEBUG("Inferring output logical tensors");
      outputSpecs_ = specializeOutputSpecs(partition_, inputSpecs_);
      GRAPH_DEBUG("Compiling partition");
      compilation_ = compile(partition_);
      is_initialized_ = true;
    }
    unlock_write();
  }

  GRAPH_DEBUG("Preparing runtime tensors");
  TensorArgs outputs;
  RunArgs runInputs, runOutputs;
  std::tie(runInputs, runOutputs) = prepareRunArgs(inputs, outputs);

  GRAPH_DEBUG("Executing partition");
  compilation_.execute(Stream::getStream(), runInputs, runOutputs);
  GRAPH_DEBUG("Partition executed");

  // Update the stack.
  drop(stack, nInputs_);
  for (auto &o : outputs)
    push_one(stack, std::move(o));
  GRAPH_DEBUG("Stack updated");
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch