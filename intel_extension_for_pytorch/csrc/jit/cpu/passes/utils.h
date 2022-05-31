#pragma once
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch {
namespace jit {
namespace graph_rewrite {
namespace utils {

struct PostOp {
  std::string ipex_op_name;
  std::vector<torch::jit::MatchFilter> filters = {};
};

const std::map<std::string, PostOp>& supported_unary_post_op_fusion_set();

// Check if the memory format of the tensor is ChannelsLast(3d)
bool is_channelslast(c10::TensorType tensor);
// Check if the memory format of the tensor is Contiguous
bool is_contiguous(c10::TensorTypePtr tensor);

} // namespace utils
} // namespace graph_rewrite
} // namespace jit
} // namespace torch
