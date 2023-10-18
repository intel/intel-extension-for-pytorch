#include <omp.h>

#include "graph_helper.h"
#include "kernel.h"
#include "operator.h"
#include "runtime.h"

#include <ATen/core/functional.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;
using namespace dnnl::graph;

using data_type = dnnl::graph::logical_tensor::data_type;

thread_local std::list<LlgaKernel::key_value_pair_t>
    LlgaKernel::cache_items_list_;
thread_local std::
    unordered_map<std::vector<int64_t>, LlgaKernel::list_iterator_t>
        LlgaKernel::cache_items_map_;
thread_local int LlgaKernel::capacity_ = 7500;

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
  nPartitionInputs_ = partition_.get_input_ports().size();
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
  for (auto& lt : partition_.get_input_ports()) {
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

ArgSpecs LlgaKernel::initializeInputSpecs(const TensorArgs& inputs) {
  ArgSpecs inputSpecs;
  inputSpecs.reserve(nPartitionInputs_);
  std::call_once(tracedInputShapesInitialized_, [&]() {
    auto numInputs = inputs.size();
    for (size_t i = 0; i < numInputs; i++) {
      tracedInputShapes_.push_back(inputs[i].sizes().vec());
      tracedInputStrides_.push_back(inputs[i].strides().vec());
    }
  });
  GRAPH_DEBUG("Initializing graph input logical tensors");
  // initializeTensorIdToOccurence can also be called just once for the first
  // input shape
  std::map<size_t, int64_t> tensorIdToOccurence =
      initializeTensorIdToOccurence();
  for (size_t i = 0; i < nGraphInputs_; i++) {
    auto spec = ArgSpec(graph_->inputs()[i]).supplementTensorInfo(inputs[i]);
    int64_t occurence = tensorIdToOccurence[spec.tid()];
    inputSpecs.insert(inputSpecs.end(), occurence, spec);
  }

  std::call_once(constantSpecInitializedFlag_, [&]() {
    for (size_t i = 0; i < nGraphInputs_; i++) {
      auto spec = ArgSpec(graph_->inputs()[i]).supplementTensorInfo(inputs[i]);
      int64_t occurence = tensorIdToOccurence[spec.tid()];
      initializedInputIds_.insert(spec.tid());
      runArgsIdx_.insert(runArgsIdx_.end(), occurence, i);
    }
    for (auto& lt : partition_.get_input_ports()) {
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
  });

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

ArgSpecs LlgaKernel::initializeOutputSpecs(
    const TensorArgs& inputs,
    bool convertDimsToUnknown = false) {
  ArgSpecs outputSpecs;
  outputSpecs.reserve(nOutputs_);

  if (convertDimsToUnknown == false) {
    auto numInputs = inputs.size();
    for (auto i = 0; i < numInputs; i++) {
      if (!((inputs[i].sizes().vec() == tracedInputShapes_[i]) &&
            (inputs[i].strides().vec() == tracedInputStrides_[i]))) {
        convertDimsToUnknown = true;
        break;
      }
    }
  }

  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = ArgSpec(graph_->outputs()[i]);
    if (convertDimsToUnknown) {
      spec = spec.convertDimsToUnknown();
    }

    if (spec.is_quantized())
      spec = getQuantizedSpec(spec, i);

    if (useOpaqueLayout(i))
      spec = spec.any();
    outputSpecs.emplace_back(spec);
  }
  return outputSpecs;
}

bool LlgaKernel::inputValueIsNotUsedLater(size_t offset) const {
  return LlgaNodeWrapper(fusionNode_).inputValueIsNotUsedLater(offset);
}

void LlgaKernel::prepareAndCacheRunArgs(
    RunArgs& runInputs,
    RunArgs& runOutputs,
    const TensorArgs& inputs,
    TensorArgs& outputs,
    ArgSpecs& inputSpecs,
    ArgSpecs& outputSpecs) {
  auto sizeOfRunArgsIdx = runArgsIdx_.size();
  auto numOfConstantInputs = constantInputs_.size();
  runInputs.reserve(sizeOfRunArgsIdx + numOfConstantInputs);
  runOutputs.reserve(nOutputs_);

  for (size_t i = 0; i < sizeOfRunArgsIdx; i++) {
    auto& spec = inputSpecs[i];
    auto& input = inputs[runArgsIdx_[i]];
    runInputs.push_back(
        {spec.logical_tensor(), Engine::getEngine(), input.data_ptr()});
  }

  for (size_t i = 0; i < numOfConstantInputs; i++) {
    // constantInputSpecs are placed after graphInputSpecs
    auto constantInputSpecIdx = nGraphInputs_ + i;
    auto& constantInputSpec = inputSpecs[constantInputSpecIdx];
    runInputs.push_back(
        {constantInputSpec.logical_tensor(),
         Engine::getEngine(),
         constantInputs_[i].data_ptr()});
  }

  outputTensorTypes_.reserve(nOutputs_);
  std::fill(outputTensorTypes_.begin(), outputTensorTypes_.end(), undefined);
  for (size_t i = 0; i < nOutputs_; i++) {
    auto& spec = outputSpecs[i];
    auto opt = c10::TensorOptions(spec.aten_scalar_type()).device(device_);

    auto outputId = spec.tid();
    auto inputOffset = inplacePairOffsets_[i];
    if ((inputOffset != INT16_MIN) && inputValueIsNotUsedLater(inputOffset)) {
      // output reuses one of input tensors
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("Inplace computation");
#endif
      GRAPH_DEBUG("INPUT INDEX OF INPLACE PAIR IS ", inputOffset);
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
            static_cast<LlgaTensorImpl*>(inputTensor.unsafeGetTensorImpl());
        switch (dataType) {
          case data_type::f32:
          case data_type::bf16:
            inputTensor = LlgaTensorImpl::llga_to_aten_tensor(llgaImpl);
            outputTensorTypes_[i] = unquantizedInplaceCompute;
            break;
          case data_type::s8:
          case data_type::u8:
            outputTensorTypes_[i] = quantizedInplaceCompute;
            inputTensor = LlgaTensorImpl::llga_to_aten_tensor(
                llgaImpl, spec.get_quantizer());
            break;
          case data_type::s32:
          default:
            TORCH_CHECK(
                false, "Invalid data type ", static_cast<size_t>(dataType));
        }
      } else {
        outputTensorTypes_[i] = unwrappedInplaceCompute;
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
      auto tensor = empty_llga(spec, opt);
      outputs.push_back(tensor);
      runOutputs.push_back(llga_from_aten_tensor(tensor));
      outputTensorTypes_[i] = betweenPartitions;
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
        outputTensorTypes_[i] = quantizedInputToFW;
      } else {
        auto tensor = at::empty_strided(spec.sizes(), spec.strides(), opt);
        outputs.push_back(tensor);
        runOutputs.push_back(
            {spec.logical_tensor(), Engine::getEngine(), tensor.data_ptr()});
        outputTensorTypes_[i] = unquantizedInputToFW;
      }
    }
  }
  TORCH_CHECK(
      std::find(
          outputTensorTypes_.begin(), outputTensorTypes_.end(), undefined) ==
          outputTensorTypes_.end(),
      "outputTensorTypes_ elements should not be undefined");
}

void LlgaKernel::prepareRunArgs(
    RunArgs& runInputs,
    RunArgs& runOutputs,
    const TensorArgs& inputs,
    TensorArgs& outputs,
    ArgSpecs& outputSpecs) {
  auto sizeOfRunArgsIdx = runArgsIdx_.size();
  for (size_t i = 0; i < sizeOfRunArgsIdx; i++) {
    auto& input = inputs[runArgsIdx_[i]];
    runInputs[i].set_data_handle(input.data_ptr());
  }

  for (size_t i = 0; i < nOutputs_; i++) {
    auto typeOfOutput = static_cast<int64_t>(outputTensorTypes_[i]);
    auto& spec = outputSpecs[i];
    auto opt = c10::TensorOptions(spec.aten_scalar_type()).device(device_);

    switch (typeOfOutput) {
      case unwrappedInplaceCompute: {
        auto inputTensor = inputs[inplacePairOffsets_[i]];
        runOutputs[i].set_data_handle(inputTensor.data_ptr());
        outputs.push_back(std::move(inputTensor));
        break;
      }
      case quantizedInplaceCompute: {
        auto inputTensor = inputs[inplacePairOffsets_[i]];
        auto llgaImpl =
            static_cast<LlgaTensorImpl*>(inputTensor.unsafeGetTensorImpl());
        inputTensor =
            LlgaTensorImpl::llga_to_aten_tensor(llgaImpl, spec.get_quantizer());
        runOutputs[i].set_data_handle(inputTensor.data_ptr());
        outputs.push_back(std::move(inputTensor));
        break;
      }
      case unquantizedInplaceCompute: {
        auto inputTensor = inputs[inplacePairOffsets_[i]];
        auto llgaImpl =
            static_cast<LlgaTensorImpl*>(inputTensor.unsafeGetTensorImpl());
        inputTensor = LlgaTensorImpl::llga_to_aten_tensor(llgaImpl);
        runOutputs[i].set_data_handle(inputTensor.data_ptr());
        outputs.push_back(std::move(inputTensor));
        break;
      }
      case betweenPartitions: {
        outputs.emplace_back(empty_llga(spec, opt));
        runOutputs[i].set_data_handle(outputs[i].data_ptr());
        break;
      }
      case quantizedInputToFW: {
        at::QuantizerPtr quantizer = spec.get_quantizer();
        outputs.emplace_back(at::new_qtensor(spec.sizes(), opt, quantizer)
                                 .as_strided_(spec.sizes(), spec.strides()));
        runOutputs[i].set_data_handle(outputs[i].data_ptr());
        break;
      }
      case unquantizedInputToFW: {
        outputs.emplace_back(
            at::empty_strided(spec.sizes(), spec.strides(), opt));
        runOutputs[i].set_data_handle(outputs[i].data_ptr());
        break;
      }
    }
  }
}

std::pair<compiled_partition, ArgSpecs> LlgaKernel::compile(
    const partition& partition,
    const TensorArgs& inputs,
    ArgSpecs& inputSpecs) {
  RECORD_FUNCTION("LLGA_bridge::compileKernel", c10::ArrayRef<c10::IValue>({}));
  auto inputLogicalTensors = fmap(inputSpecs, toLogicalTensor);
  auto outputSpecs = initializeOutputSpecs(inputs);
  auto outputLogicalTensors = fmap(outputSpecs, toLogicalTensor);
  compiled_partition compilation;
  try {
    compilation = partition.compile(
        inputLogicalTensors, outputLogicalTensors, Engine::getEngine());
  } catch (std::exception& e) {
    // if partition compilation failed, check if outputSpecs had concrete shapes
    // & strides
    bool concreteOutputStrides = false;
    for (auto outputSpec : outputSpecs) {
      if (outputSpec.sizes().size()) {
        if (outputSpec.sizes()[0] != INT64_MIN) {
          concreteOutputStrides = true;
          break;
        }
      }
    }
    if (concreteOutputStrides) {
      // recompute outputSpecs and output logical tensors
      // with INT64_MIN sizes & strides
      outputSpecs = initializeOutputSpecs(inputs, true);
      outputLogicalTensors = fmap(outputSpecs, toLogicalTensor);
      compilation = partition.compile(
          inputLogicalTensors, outputLogicalTensors, Engine::getEngine());
    } else {
      // there's nothing we can do
      throw;
    }
  }

  // Since layouts of opaque outputs would be known after compilation,
  // we need to query them out from compilation and update outputSpecs
  for (size_t i = 0; i < nOutputs_; i++) {
    auto tid = outputSpecs[i].tid();
    outputSpecs[i] =
        outputSpecs[i].update_desc(compilation.query_logical_tensor(tid));
  }

  inplacePairOffsets_.resize(nOutputs_);
  std::fill(inplacePairOffsets_.begin(), inplacePairOffsets_.end(), INT16_MIN);

  // Build static mapping from output offset to input offset
  // in accordance with available inplace options
  for (auto&& option : compilation.get_inplace_ports()) {
    size_t inputId = option.first;
    size_t outputId = option.second;
    auto inputSpecIter =
        std::find_if(inputSpecs.begin(), inputSpecs.end(), [&](auto& spec) {
          return spec.tid() == inputId;
        });
    TORCH_CHECK(inputSpecIter != inputSpecs.end(), "In-place input not found");
    auto inputOffset = inputSpecIter - inputSpecs.begin();
    auto outputSpecIter =
        std::find_if(outputSpecs.begin(), outputSpecs.end(), [&](auto& spec) {
          return spec.tid() == outputId;
        });
    TORCH_CHECK(
        outputSpecIter != outputSpecs.end(), "In-place output not found");
    auto outputOffset = outputSpecIter - outputSpecs.begin();
    inplacePairOffsets_[outputOffset] = inputOffset;
  }

  return std::make_pair(compilation, outputSpecs);
}

LlgaKernel::cp_entry& LlgaKernel::compileAndCache(
    Stack& stack,
    TensorArgs& outputs) {
  RECORD_FUNCTION("LLGA_bridge::prepareKernel", c10::ArrayRef<c10::IValue>({}));
  // Grab input values from stack
  auto stackInputs = last(stack, nGraphInputs_);
  auto inputs = fmap(stackInputs, [&](const IValue& v) {
    TORCH_CHECK(
        v.isTensor(), "Stack values for LLGA partition must be Tensor type");
    return v.toTensor();
  });
  std::vector<int64_t> key;
  key.reserve(1024);
  key.push_back(omp_get_max_threads());
  // fusionNode_ may be reassigned to another LlgaFusionGroup after ~LlgaKernel
  // would be called, and another LlgaFusionGroup may be created for another
  // graph. But since JIT graphs have had a memory leak issue for years now,
  // torch::jit::Graph::~Graph is not called after a model is traced.
  // So we would use 2 pieces of info that make a partition unique.
  key.push_back((uintptr_t)((void*)fusionNode_));
  key.push_back((uintptr_t)((void*)graph_.get()));
  for (auto& in : inputs) {
    auto shape_vec = in.sizes().vec();
    key.insert(key.end(), shape_vec.begin(), shape_vec.end());
  }
  auto iter = cache_items_map_.find(key);
  if (iter == cache_items_map_.end()) {
    GRAPH_DEBUG("Compiling partition");
    cp_entry compiledPartitionEntry;
    auto input_shape = inputs[0].sizes().vec();
    auto inputSpecs = initializeInputSpecs(inputs);
    auto compilationOutput = compile(partition_, inputs, inputSpecs);
    compiledPartitionEntry.outputSpecs_ = std::move(compilationOutput.second);
    compiledPartitionEntry.cp_ = std::move(compilationOutput.first);
    prepareAndCacheRunArgs(
        compiledPartitionEntry.inputLLGATensors_,
        compiledPartitionEntry.outputLLGATensors_,
        inputs,
        outputs,
        inputSpecs,
        compiledPartitionEntry.outputSpecs_);
    cache_items_list_.push_front(
        key_value_pair_t(key, std::move(compiledPartitionEntry)));
    cache_items_map_[key] = cache_items_list_.begin();
    if (cache_items_map_.size() > capacity_) {
      auto last = cache_items_list_.end();
      last--;
      cache_items_map_.erase(last->first);
      cache_items_list_.pop_back();
    }
    // If hash computation cost is higher than copying this struct,
    // then remove std::move above & return compiledPartitionEntry instead
    return cache_items_map_[key]->second;
  } else {
#ifdef GRAPH_DEBUG_ENABLED
    GRAPH_DEBUG("Cached compiled partition is available");
#endif
    cache_items_list_.splice(
        cache_items_list_.begin(), cache_items_list_, iter->second);
    prepareRunArgs(
        iter->second->second.inputLLGATensors_,
        iter->second->second.outputLLGATensors_,
        inputs,
        outputs,
        iter->second->second.outputSpecs_);
    return iter->second->second;
  }
}

void LlgaKernel::run(Stack& stack) {
  GRAPH_DEBUG("In ", debugName(), "\n");
  TensorArgs outputs;
  outputs.reserve(nOutputs_);

  auto& compiledPartitionEntry = compileAndCache(stack, outputs);

#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Executing partition");
#endif
  compiledPartitionEntry.cp_.execute(
      Stream::getStream(),
      compiledPartitionEntry.inputLLGATensors_,
      compiledPartitionEntry.outputLLGATensors_);

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
} // namespace torch_ipex
