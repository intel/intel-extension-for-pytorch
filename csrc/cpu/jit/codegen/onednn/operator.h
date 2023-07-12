#pragma once

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/ir/ir.h>
#include "codegen/LlgaTensorImpl.h"

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

class Operator {
 public:
  Operator(const torch::jit::Node* node, dnnl::graph::op::kind kind)
      : n(node), o(getId(node), kind, node->kind().toQualString()), k(kind) {}

  Operator& setInputValue(torch::jit::Value* v) {
    if (v->mustNotBeNone()) {
      if ((v->node()->kind() != torch::jit::prim::Constant)) {
        o.add_input(logicalTensorWithUndefSizesStrides(v));
      } else {
        o.add_input(createLogicalTensor(v));
      }
    }
    return *this;
  }

  Operator& setInput(size_t offset) {
    return setInputValue(n->input(offset));
  }

  template <typename... Ts>
  Operator& setInput(size_t offset, Ts... other) {
    setInput(offset);
    return setInput(other...);
  }

  Operator& setOutputValue(torch::jit::Value* v) {
    if (v->mustNotBeNone()) {
      if (v->node()->kind() != torch::jit::prim::Constant) {
        o.add_output(logicalTensorWithUndefSizesStrides(v));
      } else {
        o.add_output(createLogicalTensor(v));
      }
    }
    return *this;
  }

  Operator& setOutput(size_t offset) {
    return setOutputValue(n->output(offset));
  }

  template <typename... Ts>
  Operator& setOutput(size_t offset, Ts... other) {
    setOutput(offset);
    return setOutput(other...);
  }

  template <typename Attr>
  Operator& setAttr(dnnl::graph::op::attr name, Attr&& attr) {
    o.set_attr(name, std::forward<Attr>(attr));
    return *this;
  }

  template <typename F>
  Operator& setAttr(dnnl::graph::op::attr name, const F& fn, size_t offset) {
    return setAttr(name, fn(n, offset));
  }

  static std::vector<int64_t> Ints(
      const torch::jit::Node* node,
      size_t offset) {
    return torch::jit::toIValue(node->input(offset))->toIntVector();
  }

  static int64_t Int(const torch::jit::Node* node, size_t offset) {
    if (node->input(offset)->type()->isSubtypeOf(
            torch::jit::TensorType::get())) {
      // Composing FX with JIT tracing may cause scale/zps to be 0-dim tensors
      return toIValue(node->input(offset)).value().toTensor().item().toInt();
    } else {
      return static_cast<int64_t>(toIValue(node->input(offset))->toInt());
    }
  }

  static float Float(const torch::jit::Node* node, size_t offset) {
    if (node->input(offset)->type()->isSubtypeOf(
            torch::jit::TensorType::get())) {
      // Composing FX with JIT tracing may cause scale/zps to be 0-dim tensors
      return toIValue(node->input(offset)).value().toTensor().item().toFloat();
    } else {
      return static_cast<float>(toIValue(node->input(offset))->toDouble());
    }
  }

  static float ScalarToFloat(const torch::jit::Node* node, size_t offset) {
    return torch::jit::toIValue(node->input(offset))->toScalar().to<float>();
  }

  static std::vector<float> FloatValueToVector(float value) {
    return {value};
  }

  static std::vector<float> FloatToVector(
      const torch::jit::Node* node,
      size_t offset) {
    return FloatValueToVector(Float(node, offset));
  }

  static std::vector<int64_t> IntValueToVector(int64_t value) {
    return {value};
  }

  static std::vector<int64_t> IntToVector(
      const torch::jit::Node* node,
      size_t offset) {
    return IntValueToVector(Int(node, offset));
  }

  static std::string QuantString(at::ScalarType scalar_type) {
    switch (scalar_type) {
      case at::ScalarType::QInt8:
        return std::string("int8");
      case at::ScalarType::QUInt8:
        return std::string("uint8");
      default:
        TORCH_CHECK(
            false,
            "Invalid quant data type ",
            static_cast<size_t>(scalar_type));
    }
  }

  static std::string String(const torch::jit::Node* node, size_t offset) {
    return QuantString(static_cast<at::ScalarType>(Int(node, offset)));
  }

  static at::Tensor Tensor(const torch::jit::Node* node, size_t offset) {
    return torch::jit::toIValue(node->input(offset))->toTensor();
  }

  static bool Bool(const torch::jit::Node* node, size_t offset) {
    return torch::jit::toIValue(node->input(offset))->toBool();
  }

  static uint64_t getId(const torch::jit::Node* node) {
    return reinterpret_cast<uint64_t>(node); // cast node address as op id
  }

  static torch::jit::Node* getNode(uint64_t opId) {
    return reinterpret_cast<torch::jit::Node*>(opId);
  }

  dnnl::graph::op::kind kind() const {
    return k;
  }

  dnnl::graph::op llgaOp() const {
    return o;
  }

 private:
  dnnl::graph::logical_tensor createLogicalTensor(
      torch::jit::Value* value) const {
    return LlgaTensorDesc(value).logical_tensor();
  }

  // We use shapes & strides as -1 for outputs of ops that we'd try mapping to
  // LLGA
  dnnl::graph::logical_tensor logicalTensorWithUndefSizesStrides(
      torch::jit::Value* value) const {
    return LlgaTensorDesc(value).convertDimsToUnknown().logical_tensor();
  }

  const torch::jit::Node* n;
  dnnl::graph::op o;
  dnnl::graph::op::kind k;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
