#include "graph_ext.h"
#include "accelerated_ops.h"

namespace torch { namespace jit {
void NodeExt::initFormatInfo() {
  std::vector<int64_t> formatInfo (
      this->inputs().size() + this->outputs().size(),
      formatTag::any);

  this->is_(attr::format_info, std::move(formatInfo));
}

const std::vector<int64_t>& NodeExt::getFormatInfo() const {
  return this->is(attr::format_info);
}

formatTag NodeExt::inputFormat(int i) const {
  return static_cast<formatTag>(getFormatInfo().at(i));
}

formatTag NodeExt::outputFormat(int i) const {
  return static_cast<formatTag>(
      getFormatInfo().at(this->inputs().size() + i));
}

void NodeExt::setInputFormat(formatTag format, int i) {
  auto& formatInfo = const_cast<std::vector<int64_t>&>(getFormatInfo());
  formatInfo.at(i) = static_cast<int64_t>(format);
}

void NodeExt::setOutputFormat(formatTag format, int i) {
  setInputFormat(format, this->inputs().size() + i);
}

int64_t NodeExt::getGroupInfo() const {
  return this->i(attr::group_info);
};

void NodeExt::setGroupInfo(int64_t groups) {
  this->i_(attr::group_info, groups);
}

Node *NodeExt::createReorder(Value *v, Graph *g, formatTag from, formatTag to) {
  NodeExt *reorder = nullptr;
  if (from != to) {
    reorder = reinterpret_cast<NodeExt *>(g->create(dnnl::reorder));
    reorder->output()->setDebugName(v->debugName() + ".reorder");
    reorder->output()->setType(v->type());
  }

  return reorder;
}

Node* NodeExt::insertReorder(
    Value *v, Node *insert_point, formatTag from, formatTag to) {
  auto *reorder = createReorder(v, insert_point->owningGraph(), from , to);

  if (reorder != nullptr) {
    WithInsertPoint guard(insert_point);
    reorder->insertAfter(insert_point);
  }

  return reorder;
}

void NodeExt::prependReorders(
    use_list uses, formatList froms, groupsList groups) {
  for (int i = 0; i < uses.size(); ++ i) {
    auto u = uses[i];
    auto *node = reinterpret_cast<NodeExt *>(u.user);
    auto from = froms.at(i);
    auto group = groups.at(i);
    auto *v = u.user->inputs().at(u.offset);

    auto to = node->inputFormat(u.offset);
    auto* reorder = reinterpret_cast<NodeExt*>(
        insertReorder(v, v->node(), from, to));

    if (reorder != nullptr) {
      u.user->replaceInput(u.offset, reorder->output());
      reorder->addInput(v);
      // We have inputs/outputs information correctly
      reorder->initFormatInfo();
      reorder->setGroupInfo(group);
      reorder->setInputFormat(from);
      reorder->setOutputFormat(to);
    }
  }
}

Node* NodeExt::appendReorder(formatTag to, int i) {
  auto *v = this->outputs().at(i);
  auto from = this->outputFormat(i);
  auto reorder = reinterpret_cast<NodeExt*>(insertReorder(v, this, from, to));

  if (reorder != nullptr) {
    v->replaceAllUsesWith(reorder->output());
    reorder->addInput(v);

    // We have inputs/outputs information correctly
    reorder->initFormatInfo();
    reorder->setGroupInfo(1);
    reorder->setInputFormat(from);
    reorder->setOutputFormat(to);
  }

  return reorder;
}

void NodeExt::propagateFormats() {
  // TODO: Need consultant with acceleration libraries
  setOutputFormat(inputFormat());
}

bool NodeExt::isDNNLOps() const {
  return std::string(this->kind().ns().toQualString()) == "namespaces::dnnl";
}

// XXX: ??? Do we duplicated all the info ???
Node* replaceOpWithNewKind(Node *node, Graph *g, NodeKind kind) {
  auto newNode = g->create(kind);
  newNode->insertBefore(node);
  newNode->setScope(node->scope());
  newNode->copyAttributes(*node);

  for (auto input : node->inputs()) {
    newNode->addInput(input);
  }

  //
  // XXX: currently we substitue all single output node
  //
  newNode->output()->copyMetadata(node->output());
  newNode->output()->setType(node->output()->type());
  node->replaceAllUsesWith(newNode);
  node->destroy();

  return newNode;
}

Node* fuseOpsWithNewKind(Node *curr, Value *v, Graph *g, NodeKind kind) {
  auto newNode = g->create(kind);
  auto prev = v->node();
  newNode->insertBefore(prev);
  newNode->setScope(prev->scope());
  newNode->copyAttributes(*prev);

  for (auto input : prev->inputs()) {
    newNode->addInput(input);
  }

  for (auto input : curr->inputs()) {
    if (input != v) {
      newNode->addInput(input);
    }
  }

  // Copy curr or prev?
  newNode->output()->copyMetadata(prev->output());
  newNode->output()->setType(prev->output()->type());

  v->replaceAllUsesWith(newNode->output());
  curr->replaceAllUsesWith(newNode);

  prev->destroy();
  curr->destroy();

  return newNode;
}

bool Conv2dNode::couldInferFormats() const {
  return (this->input(1)->node()->kind() == prim::Constant
      // && this->input(2)->isCompleteTensor()
      && this->input(3)->node()->kind() == prim::Constant
      && this->input(4)->node()->kind() == prim::Constant
      && this->input(5)->node()->kind() == prim::Constant
      && this->input(6)->node()->kind() == prim::Constant);
}

bool Conv2dNode::hasConstantParams() const {
  bool has = true;
  for (int i = 1; i < inputs().size(); ++i) {
    has = has && (this->input(i)->node()->kind() == prim::Constant);
  }

  return has;
}

formatTag Conv2dNode::expectedWeightFormat(
    c10::ArrayRef<int64_t> sizes,
    c10::List<int64_t> stride,
    c10::List<int64_t> padding,
    c10::List<int64_t> dilation,
    int64_t groups, dataType dtype) const {
  tensor::dims weight_dims (sizes.begin(), sizes.end());
  tensor::dims strides (stride.begin(), stride.end());
  tensor::dims padding_l {(int)padding[0], (int)padding[0]};
  tensor::dims padding_r {(int)padding[1], (int)padding[1]};
  tensor::dims dilates (dilation.size());
  std::transform(dilation.begin(), dilation.end(), dilates.begin(),
      [](int64_t d) { return (int)(d -1); });

  // We don't consider winograd
  auto desc = ideep::convolution_forward::expected_weights_descriptor(
      weight_dims, dtype, strides, padding_l, padding_r, dilates, groups);

  return desc.get_internal_format();
}

void Conv2dNode::fixWeightFormatIfPossible() {
  if (couldInferFormats()) {
    auto tensor = toIValue(this->input(1))->toTensor();
    auto sizes = tensor.sizes();
    auto stride = toIValue(this->input(3))->toIntList();
    auto padding = toIValue(this->input(4))->toIntList();
    auto dilation = toIValue(this->input(5))->toIntList();
    auto groups = toIValue(this->input(6))->toInt();
    auto wformat = expectedWeightFormat(sizes, std::move(stride),
        std::move(padding), std::move(dilation), groups);
    this->setInputFormat(wformat, 1);
    this->prependReorders(use_list {{this, 1}}, {natureWeightFormat}, {groups});
  }
}

bool BatchNorm2dNode::hasConstantParams() const {
  bool has =
    this->input(1)->node()->kind() == prim::Constant
      && this->input(2)->node()->kind() == prim::Constant
      && this->input(3)->node()->kind() == prim::Constant
      && this->input(4)->node()->kind() == prim::Constant
      && this->input(7)->node()->kind() == prim::Constant;

  // TODO: more check to make sure

  return has;
}

}} // namespace torch::jit
