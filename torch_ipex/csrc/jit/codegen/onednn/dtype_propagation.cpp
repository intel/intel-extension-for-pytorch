#include "jit/codegen/onednn/graph_helper.h"
#include "jit/codegen/onednn/dtype_propagation.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

static std::map<std::string, OutputDtype> output_dtype {
    {std::string("uint8"), OutputDtype::uint8},
    {std::string("int8"), OutputDtype::int8},
    {std::string("fp32"), OutputDtype::fp32},
    {std::string("default"), OutputDtype::undef},
};

void DtypePropagation(Node* n) {
  if (!LlgaGraphHelper::isLlgaSubgraph(n))
    return;
  
  // TODO: only handle 1 output for now
  if (n->outputs().size() != 1)
    return;

  size_t offset = 0;

  auto subgraph = n->g(attr::Subgraph);

  auto output_node = subgraph->outputs()[0]->node();

  if (output_node->kind() == Symbol::aten("quantize_per_tensor")) {
    LlgaNodeWrapper(n).setOutputDtypes(offset, output_dtype[Operator::String(output_node, /* offset */3)]);
  } else if (output_node->kind() == Symbol::aten("quantize_per_channel")) {
    LlgaNodeWrapper(n).setOutputDtypes(offset, output_dtype[Operator::String(output_node, /* offset */4)]);
  } else if (output_node->kind() == Symbol::aten("dequantize")) {
    LlgaNodeWrapper(n).setOutputDtypes(offset, output_dtype[std::string("fp32")]);
  } else {
    LlgaNodeWrapper(n).setOutputDtypes(offset, output_dtype[std::string("default")]);
  }
  return;
}

void DtypePropagation(at::ArrayRef<Block*> blocks) {
  for (Block* block : blocks)
    for (Node* node : block->nodes())
      DtypePropagation(node);
}

void PropagateDtype(const std::shared_ptr<Graph>& graph) {
  DtypePropagation(graph->block());
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch