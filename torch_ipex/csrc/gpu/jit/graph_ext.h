#pragma once

#include <vector>
#include <memory>
#include <dnnl.hpp>
#include "accelerated_ops.h"
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/constants.h>

namespace torch { namespace jit {
using namespace dnnl;
using dataType = dnnl::memory::data_type;
using formatTag = dnnl::memory::format_tag;
using formatList = std::vector<formatTag>;
using groupsList = std::vector<int64_t>;

static constexpr auto natureFormat = formatTag::nchw;
static constexpr auto natureWeightFormat = formatTag::oihw;

// attributes for pyrys ops to decide which format is on
// Or what formats transfered by reorder
// ints type
namespace attr {
// Integer string format
static auto format_info = Symbol::attr("format_info");
static auto block_info = Symbol::attr("block_info");
static auto group_info = Symbol::attr("group_info");
}

//
// A convenient wrapper to extend op to support formats and stuff
// No construcion is allowed, pointer only
//
class NodeExt : public Node {
public:
  NodeExt() = delete;
public:
  // Old API support
  // -------------------------------------------------------------------
  // Get format information from node
  formatTag inputFormat(int i = 0) const;
  formatTag outputFormat(int i = 0) const;

  // Set format information in node
  void setInputFormat(formatTag format, int i = 0);
  void setOutputFormat(formatTag format, int i = 0);

  int64_t getGroupInfo() const;
  void setGroupInfo(int64_t groups);

  // Add reorder to inputs
  void prependReorders(use_list uses, formatList froms, groupsList groups);

  // append reorder at output index i
  Node *appendReorder(formatTag to, int i = 0);

  // Op Format protocol, adjustFormats according to inputs???
  void propagateFormats();

  #if 0
  bool isReorder() const {
    return this->kind() == dpcpp::reorder;
  }
  #endif

  bool isDNNLOps () const ;

  #if 0
  bool isConv2d () const {
    return this->kind() == dpcpp::conv2d;
  }
  bool isBatchNorm () const {
    return this->kind() == dpcpp::batch_norm;
  }
  #endif

  void initFormatInfo();

  template <class T> T* cast() {
    return reinterpret_cast<T*>(this);
  }
private:
  // we save formats as Ints attribute internally
  const std::vector<int64_t>& getFormatInfo() const;

  static Node* createReorder(
      Value *v, Graph *g, formatTag from, formatTag to);
  static Node* insertReorder(
      Value *v, Node *insert_point, formatTag from, formatTag to);
};

class Conv2dNode : public NodeExt {
public:
  bool couldInferFormats() const;
  bool hasConstantParams() const;
  #if 0
  void fixWeightFormatIfPossible();
  formatTag expectedWeightFormat(
      c10::ArrayRef<int64_t> sizes,
      c10::List<int64_t> stride,
      c10::List<int64_t> padding,
      c10::List<int64_t> dilation,
      int64_t groups, dataType dtype = dataType::f32) const;
  #endif
};

class BatchNorm2dNode : public NodeExt {
public:
  bool hasConstantParams() const;
};

Node *replaceOpWithNewKind(Node *old, Graph *g, NodeKind kind);
Node *fuseOpsWithNewKind(Node *curr, Value *v, Graph *g, NodeKind kind);

}} // namespace torch::jit
