#include <omp.h>

#include "kernel.h"
#include "graph_helper.h"
#include "operator.h"
#include "runtime.h"

#include <ATen/core/functional.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using namespace dnnl::graph;

using data_type = dnnl::graph::logical_tensor::data_type;

LlgaKernel::LlgaKernel(const Node* fusionNode)
    : fusionNode_(fusionNode),
      graph_(fusionNode->g(attr::Subgraph)),
      nGraphInputs_(graph_->inputs().size()),
      nOutputs_(graph_->outputs().size()),
      debugName_(genDebugName()),
      profileName_(genProfileName()) {
  // TODO: This is a workaround to recreate the partitions here.
  // The ideal way is to use the partition serialization API (not available from
  // LLGA now) to carry a serialized string representation from graph rewrite
  // and deserialize it here.
  auto llgaGraphHelper = LlgaGraphHelper(graph_);
  auto partitions = llgaGraphHelper.getPartitions();
  tensorIdToValue_ = llgaGraphHelper.getTensorIdToValue();

  TORCH_CHECK(
      partitions.size() == 1,
      "LLGA subgraph should contain only one partition");
  partition_ = partitions[0];
  nPartitionInputs_ = partition_.get_in_ports().size();
  GRAPH_DEBUG("Initialized ", debugName(), "\n", graph_->toString());
}

bool LlgaKernel::useOpaqueLayout(size_t offset) const {
  return LlgaNodeWrapper(fusionNode_).useOpaqueLayout(offset);
}

ArgSpec LlgaKernel::getQuantizedSpec(ArgSpec spec, size_t offset) const {
  auto node = graph_->outputs()[offset]->node();
  TORCH_CHECK(
      node->kind() == Symbol::aten("quantize_per_tensor") ||
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

std::map<size_t, int64_t> LlgaKernel::initializeTensorIdToOccurence() const {
  std::map<size_t, int64_t> tensorIdToOccurence;
  for (auto& lt : partition_.get_in_ports()) {
    auto inputId = lt.get_id();
    std::map<size_t, int64_t>::iterator it(tensorIdToOccurence.find(inputId));
    if (it != tensorIdToOccurence.end()) {
      it->second++;
    } else {
      tensorIdToOccurence[inputId] = 1;
    }
  }
  return tensorIdToOccurence;
}

void LlgaKernel::initializeConstantInputs() {
  for (auto& lt : partition_.get_in_ports()) {
    auto inputId = lt.get_id();
    if (initializedInputIds_.find(inputId) == initializedInputIds_.end()) {
      TORCH_CHECK(
          tensorIdToValue_.count(inputId) > 0,
          "inputs with inputId ",
          inputId,
          " is missing");
      auto* value = tensorIdToValue_[inputId];

      TORCH_CHECK(
          value->node()->kind() == prim::Constant &&
              value->type()->cast<TensorType>(),
          "inputs with inputId ",
          inputId,
          " should be a Constant tensor");
      constantValues_.emplace_back(value);

      auto const_tensor = toIValue(value)->toTensor();
      constantInputs_.emplace_back(const_tensor);
    }
  }
}

ArgSpecs LlgaKernel::initializeInputSpecs(const TensorArgs& inputs) {
  ArgSpecs inputSpecs;
  inputSpecs.reserve(nPartitionInputs_);
  GRAPH_DEBUG("Initializing graph input logical tensors");

  std::map<size_t, int64_t> tensorIdToOccurence =
      initializeTensorIdToOccurence();
  for (size_t i = 0; i < nGraphInputs_; i++) {
    auto spec = ArgSpec(graph_->inputs()[i]).supplementTensorInfo(inputs[i]);
    initializedInputIds_.insert(spec.tid());

    int64_t occurence = tensorIdToOccurence[spec.tid()];
    inputSpecs.insert(inputSpecs.end(), occurence, spec);
    runArgsIdx_.insert(runArgsIdx_.end(), occurence, i);
  }

  GRAPH_DEBUG("Initializing constant input tensors");
  initializeConstantInputs();

  TORCH_CHECK(
      inputSpecs.size() + constantValues_.size() == nPartitionInputs_,
      "Partition inputs are missing");

  GRAPH_DEBUG(
      "Concatenating constant input logical tensors to graph input "
      "logical tensors");
  for (size_t i = 0; i < constantValues_.size(); i++) {
    inputSpecs.emplace_back(ArgSpec(constantValues_[i]));
  }
  return inputSpecs;
}

ArgSpecs LlgaKernel::initializeOutputSpecs() const {
  ArgSpecs outputSpecs;
  outputSpecs.reserve(nOutputs_);
  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = ArgSpec(graph_->outputs()[i]);

    if (spec.is_quantized())
      spec = getQuantizedSpec(spec, i);

    if (useOpaqueLayout(i))
      spec = spec.any();
    outputSpecs.emplace_back(spec);
  }
  return outputSpecs;
}

std::tuple<RunArgs, RunArgs> LlgaKernel::prepareRunArgs(
    const TensorArgs& inputs,
    TensorArgs& outputs) const {
  RECORD_FUNCTION(
      "LLGA_bridge::prepareRunArgs", c10::ArrayRef<c10::IValue>({}));

  RunArgs runInputs, runOutputs;
  for (size_t i = 0; i < runArgsIdx_.size(); i++) {
    auto spec = inputSpecs_[i];
    auto input = inputs[runArgsIdx_[i]];
    runInputs.push_back(
        {spec.logical_tensor(), Engine::getEngine(), input.data_ptr()});
  }
  for (size_t i = 0; i < constantInputs_.size(); i++) {
    // constantInputSpecs are placed after graphInputSpecs
    auto constantInputSpecIdx = nGraphInputs_ + i;
    auto constantInputSpec = inputSpecs_[constantInputSpecIdx];
    runInputs.push_back(
        {constantInputSpec.logical_tensor(),
         Engine::getEngine(),
         constantInputs_[i].data_ptr()});
  }

  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = outputSpecs_[i];
    auto opt = c10::TensorOptions(spec.aten_scalar_type()).device(device_);

    auto outputId = spec.tid();
    auto iter = inplacePairs_.find(outputId);
    if (iter != inplacePairs_.end()) {
      // output reuses one of input tensors
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("Inplace computation");
#endif
      auto inputOffset = iter->second;
      auto inputTensor = inputs[inputOffset];
      auto dataType = spec.dtype();
      if (C10_UNLIKELY(!useOpaqueLayout(i) && inputTensor.is_mkldnn())) {
        // If the input tensor was between two partitions, it would've been
        // wrapped with LlgaTensorImpl. But if it's being reused as the output
        // tensor, which is not between two partitions, then we'd have to
        // re-wrap it with a sub-class of TensorImpl, as it'd be fed into a
        // PyTorch op.
#ifdef GRAPH_DEBUG_ENABLED
        GRAPH_DEBUG("Rewrap tensor");
#endif
        auto llgaImpl =
            static_cast<at::LlgaTensorImpl*>(inputTensor.unsafeGetTensorImpl());
        switch (dataType) {
          case data_type::f32:
          case data_type::bf16:
            inputTensor = at::LlgaTensorImpl::llga_to_aten_tensor(llgaImpl);
            break;
          case data_type::s8:
          case data_type::u8:
            inputTensor = at::LlgaTensorImpl::llga_to_aten_tensor(
                llgaImpl, spec.get_quantizer());
            break;
          case data_type::s32:
          default:
            TORCH_CHECK(
                false, "Invalid data type ", static_cast<size_t>(dataType));
        }
      }
      outputs.push_back(inputTensor);
      runOutputs.push_back(
          {spec.logical_tensor(), Engine::getEngine(), inputTensor.data_ptr()});
    } else if (useOpaqueLayout(i)) {
      // Wrap tensors between partitions with LlgaTensorImpl wrapper, so that we
      // can bypass guard-check, as strides would be different than those
      // expected.
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("Between partitions");
#endif
      auto tensor = at::empty_llga(spec, opt);
      outputs.push_back(tensor);
      runOutputs.push_back(at::llga_from_aten_tensor(tensor));
    } else {
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("Neither opaque nor inplace");
#endif
      if (spec.is_quantized()) {
        at::QuantizerPtr quantizer = spec.get_quantizer();
        auto qtensor = at::new_qtensor(spec.sizes(), opt, quantizer);
        // TODO: Setting strides is possible only on uniformly quantized tensor.
        // Currently, only weight will use quantize_per_channel, data will
        // always use quantize_per_tensor. We will only allocate buffer for data
        // (output of a LlgaPartition). If in the future, we need allocate
        // buffer for qensor that is quantized per channel, need implemeted
        // as_strided_qtensorimpl for PER_CHANNEL QScheme.
        qtensor.as_strided_(spec.sizes(), spec.strides());
        outputs.push_back(qtensor);
        runOutputs.push_back(
            {spec.logical_tensor(), Engine::getEngine(), qtensor.data_ptr()});
      } else {
        auto tensor = at::empty_strided(spec.sizes(), spec.strides(), opt);
        outputs.push_back(tensor);
        runOutputs.push_back(
            {spec.logical_tensor(), Engine::getEngine(), tensor.data_ptr()});
      }
    }
  }

  return std::make_tuple(runInputs, runOutputs);
}

compiled_partition LlgaKernel::compile(const partition& partition) {
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
  for (auto&& option : compilation.get_inplace_ports()) {
    size_t inputId = option.first;
    size_t outputId = option.second;
    auto inputSpecIter =
        std::find_if(inputSpecs_.begin(), inputSpecs_.end(), [&](auto& spec) {
          return spec.tid() == inputId;
        });
    TORCH_CHECK(inputSpecIter != inputSpecs_.end(), "In-place input not found");
    auto inputOffset = inputSpecIter - inputSpecs_.begin();
    inplacePairs_[outputId] = inputOffset;
  }

  return compilation;
}

dnnl::graph::compiled_partition& LlgaKernel::compileAndCache(
    const dnnl::graph::partition& partition,
    int n_thread) {
  // index starts from 0 while min(omp_get_max_threads) = 1
  int i_thread = n_thread - 1;
  std::call_once(compilation_initialized_flags_[i_thread], [&]() {
    GRAPH_DEBUG("Compiling partition for i_thread ", i_thread);
    compilations_[i_thread] = compile(partition_);
  });
  return compilations_[i_thread];
}

void LlgaKernel::run(Stack& stack) {
  GRAPH_DEBUG("In ", debugName(), "\n");

  // Grab input values from stack
  auto stackInputs = last(stack, nGraphInputs_);
  auto inputs = fmap(stackInputs, [&](const IValue& v) {
    TORCH_CHECK(
        v.isTensor(), "Stack values for LLGA partition must be Tensor type");
    return v.toTensor();
  });

  // Input and output specs are not related to omp_num_threads
  std::call_once(
      spec_initialized_flag_,
      [&](const TensorArgs& inputs) {
#ifdef GRAPH_DEBUG_ENABLED
        GRAPH_DEBUG("Initializing input logical tensors");
#endif
        inputSpecs_ = initializeInputSpecs(inputs);
#ifdef GRAPH_DEBUG_ENABLED
        GRAPH_DEBUG("Initializing output logical tensors");
#endif
        outputSpecs_ = initializeOutputSpecs();
      },
      inputs);

  TensorArgs outputs;
  RunArgs runInputs, runOutputs;
  dnnl::graph::compiled_partition compilation;

  int n_thread = omp_get_max_threads();
  if (n_thread > 0 && n_thread <= MAX_COMPILATION_CACHE_SIZE) {
#ifdef GRAPH_DEBUG_ENABLED
    GRAPH_DEBUG("Cached compilation");
#endif
    compilation = compileAndCache(partition_, n_thread);
  } else {
#ifdef GRAPH_DEBUG_ENABLED
    GRAPH_DEBUG("Runtime compilation");
#endif
    compilation = compile(partition_);
  }
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Preparing runtime tensors");
#endif
  std::tie(runInputs, runOutputs) = prepareRunArgs(inputs, outputs);
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Executing partition");
#endif
  compilation.execute(Stream::getStream(), runInputs, runOutputs);
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Partition executed");
#endif
  // Update the stack.
  drop(stack, nGraphInputs_);
  for (auto& o : outputs) {
    push_one(stack, std::move(o));
  }
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Stack updated");
#endif
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
