#pragma once

#include "torch_ipex/csrc/LlgaTensorImpl.h"
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

class Operator {
 public:
  Operator(const Node* node, dnnl::graph::op::kind kind)
      : n(node), o(getId(node), kind, node->kind().toQualString()), k(kind) {
    setAttr("data_format", std::string("NCX"));
  }

  Operator& setInputValue(Value* v) {
    if (v->mustNotBeNone())
      o.add_input(createLogicalTensor(v));
    return *this;
  }

  Operator& setQuantizationInputValue(Value* v, dnnl::graph::logical_tensor::data_type dtype) {
    o.add_input(createLogicalTensor(v, dtype));
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

  Operator& setOutputValue(Value* v) {
    if (v->mustNotBeNone())
      o.add_output(createLogicalTensor(v));
    return *this;
  }

  Operator& setQuantizationOutputValue(Value* v, dnnl::graph::logical_tensor::data_type dtype) {
    o.add_output(createLogicalTensor(v, dtype));
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
  Operator& setAttr(std::string name, Attr&& attr) {
    o.set_attr(name, std::forward<Attr>(attr));
    return *this;
  }

  template <typename F>
  Operator& setAttr(std::string name, const F& fn, size_t offset) {
    return setAttr(name, fn(n, offset));
  }

  static std::vector<int64_t> Ints(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toIntVector();
  }

  static int64_t Int(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toInt();
  }

  static float Float(const Node* node, size_t offset) {
    return static_cast<float>(toIValue(node->input(offset))->toDouble());
  }

  static std::vector<float> FloatToVector(const Node* node, size_t offset) {
    return {Float(node, offset)};
  }

  static std::vector<int64_t> IntToVector(const Node* node, size_t offset) {
    return {Int(node, offset)};
  }

  // TODO: tensor to vector? We have assumed that zp and scale tensor is contiguous
  static std::vector<float> FloatTensorToVector(const Node* node, size_t offset) {
    at::Tensor tensor = toIValue(node->input(offset))->toTensor();
    std::vector<float> vectors;
    for (int i = 0; i < tensor.numel(); i++) {
      vectors.push_back(tensor[i].item().toFloat());
    }
    return vectors;
  }

  static std::vector<int64_t> IntTensorToVector(const Node* node, size_t offset) {
    at::Tensor tensor = toIValue(node->input(offset))->toTensor();
    std::vector<int64_t> vectors;
    for (int i = 0; i < tensor.numel(); i++) {
      vectors.push_back(tensor[i].item().toInt());
    }
    return vectors;
  }

  static std::string String(const Node* node, size_t offset) {
    std::string str = toString(static_cast<at::ScalarType>(Int(node, offset)));
    TORCH_CHECK((str == "QUInt8") || (str == "QInt8"), "Incorrect scalar type: ", str);
    if (str == "QUInt8") {
      return std::string("uint8");
    } else {
      return std::string("int8");
    }
  }

  static bool Bool(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toBool();
  }

  static uint64_t getId(const Node* node) {
    return reinterpret_cast<uint64_t>(node); // cast node address as op id
  }

  dnnl::graph::op::kind kind() const {
    return k;
  }

  dnnl::graph::op llgaOp() const {
    return o;
  }

 private:
  dnnl::graph::logical_tensor createLogicalTensor(Value* value) const {
    return at::LlgaTensorDesc(value).logical_tensor();
  }

  dnnl::graph::logical_tensor createLogicalTensor(Value* value, dnnl::graph::logical_tensor::data_type dtype) const {
    return at::LlgaTensorDesc(value, dtype).logical_tensor();
  }

  const Node* n;
  dnnl::graph::op o;
  dnnl::graph::op::kind k;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch