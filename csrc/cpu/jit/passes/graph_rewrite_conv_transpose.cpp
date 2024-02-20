#include <ideep.hpp>
#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"
#include "utils.h"

#include <ATen/code_template.h>

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;
using namespace torch_ipex::cpu;
using namespace torch::jit;

void insertPrePackedConvTransposeOpForATen(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      insertPrePackedConvTransposeOpForATen(block);
    }
    // TODO: add conv_transpose1d
    if (n->kind() == aten::conv_transpose2d ||
        n->kind() == aten::conv_transpose3d) {
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();
      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      // if can't get input shape info, will not do weight prepack.
      if (!(input_size_option.has_value() &&
            (input_size_option.value().size() == 4 ||
             input_size_option.value().size() == 5))) {
        continue;
      }
      IValue input_size_value(input_size_option.value());

      auto weight_tensor_type = n->inputs().at(1)->type()->cast<TensorType>();
      auto weight_size_option = weight_tensor_type->sizes().concrete_sizes();
      // weight has not shape info, will not do weight prapacked.
      if (!(weight_size_option.has_value() &&
            (weight_size_option.value().size() == 4 ||
             weight_size_option.value().size() == 5))) {
        continue;
      }
      const auto dtype = weight_tensor_type->scalarType();
      if (dtype.has_value()) {
        if (*dtype == at::ScalarType::BFloat16 &&
            !ideep::has_bf16_type_support())
          continue;
        if (*dtype == at::ScalarType::Half && !ideep::has_fp16_type_support())
          continue;
      }
      // # padding - output_padding + stride <= 0 unsupported in mkldnn
      auto stride = toIValue(n->input(3))->toIntList();
      auto padding = toIValue(n->input(4))->toIntList();
      auto output_padding = toIValue(n->input(5))->toIntList();

      auto weight_size = weight_size_option.value();
      // 2d case.
      if (weight_size.size() == 4) {
        if (padding[0] - output_padding[0] + stride[0] <= 0 ||
            padding[1] - output_padding[1] + stride[1] <= 0) {
          continue;
        }
      }
      // 3d case.
      if (weight_size.size() == 5) {
        if (padding[0] - output_padding[0] + stride[0] <= 0 ||
            padding[1] - output_padding[1] + stride[1] <= 0 ||
            padding[2] - output_padding[2] + stride[2] <= 0) {
          continue;
        }
      }

      std::vector<int64_t> k_size = {weight_size[2]};
      // 2d case.
      if (weight_size.size() == 4) {
        k_size.push_back(weight_size[3]);
      }
      // 3d case.
      if (weight_size.size() == 5) {
        k_size.push_back(weight_size[3]);
        k_size.push_back(weight_size[4]);
      }
      // w_is_channels_last is invaild, there will has a check the memory
      // format at convolution kernel side.
      bool w_is_channels_last = false;
      if (constant_as<at::Tensor>(n->namedInput("weight")).has_value()) {
        at::Tensor weight_tensor =
            constant_as<at::Tensor>(n->namedInput("weight")).value();
        w_is_channels_last =
            weight_tensor.is_contiguous(at::MemoryFormat::ChannelsLast) ||
            weight_tensor.is_contiguous(at::MemoryFormat::ChannelsLast3d);
      }
      IValue weight_is_channels_last_value(w_is_channels_last);
      auto weight_is_channels_last =
          graph->insertConstant(weight_is_channels_last_value);

      // Note that once creating this "conv_transpose_prepack" node, make sure
      // it is also inserted into the graph. Details ref to "linear_prepack"
      // creation in "graph_rewrite_linear.cpp"
      auto prepack_node = graph->create(
          Symbol::fromQualString("ipex_prepack::conv_transpose_prepack"), 1);
      for (auto i = 1; i < n->inputs().size(); ++i) {
        Value* v = n->inputs().at(i);
        prepack_node->addInput(v);
      }
      prepack_node->addInput(weight_is_channels_last);

      auto input_size = graph->insertConstant(input_size_value);
      prepack_node->addInput(input_size);
      prepack_node->output()->setType(getCustomClass(
          "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext"));

      graph->insertNode(prepack_node);
      auto prepack_conv_transpose = graph->insertNode(graph->create(
          Symbol::fromQualString("ipex_prepack::conv_transpose_run"), 1));
      prepack_conv_transpose->addInput(n->inputs().at(0));
      prepack_conv_transpose->addInput(prepack_node->output());
      prepack_conv_transpose->output()->setType(
          n->output()->type()->cast<TensorType>());
      auto v = n->outputs().at(0);
      n->output()->replaceAllUsesWith(prepack_conv_transpose->output());
    }
  }
  EliminateDeadCode(b);
}

// For ipex conv_transpose, we can re-pack the packed weight in the op-context
// if we get an different input size here
void mayRePackConvTransposeOpForIpex(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      mayRePackConvTransposeOpForIpex(block);
    }
    // TODO: add conv_transpose3d
    if (n->kind() == Symbol::fromQualString("torch_ipex::conv_transpose")) {
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();
      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      // if can't get input shape info, will not do weight prepack.
      if (!(input_size_option.has_value() &&
            (input_size_option.value().size() == 4 ||
             input_size_option.value().size() == 5))) {
        continue;
      }
      IValue input_size_value(input_size_option.value());
      auto prepack_node = n->inputs().at(3)->node()->inputs().at(0);
      // For graph before "freeze", cannot get custom class to repack
      if (!toIValue(prepack_node).has_value())
        continue;
      auto convtranspose_op_ctx = toIValue(prepack_node)
                                      .value()
                                      .toCustomClass<ConvTransposeOpContext>();
      convtranspose_op_ctx->may_repack(input_size_option.value());
      auto prepack_convtranspose = graph->insertNode(graph->create(
          Symbol::fromQualString("ipex_prepack::conv_transpose_run"), 1));
      prepack_convtranspose->addInput(n->inputs().at(0));
      prepack_convtranspose->addInput(prepack_node);
      prepack_convtranspose->output()->setType(
          n->output()->type()->cast<TensorType>());
      auto v = n->outputs().at(0);
      n->output()->replaceAllUsesWith(prepack_convtranspose->output());
    }
  }
  EliminateDeadCode(b);
}

void insertPrePackedConvTransposeOp(std::shared_ptr<Graph>& graph) {
  insertPrePackedConvTransposeOpForATen(graph->block());
  mayRePackConvTransposeOpForIpex(graph->block());
}

void fuseConvTransposeWithEltwise(std::shared_ptr<Graph>& graph) {
  // For unary post OPs:
  auto conv_transpose_op_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %packed_weight):
        %x : Tensor = ipex_prepack::conv_transpose_run(%input, %packed_weight)
        %res = ${op}(%x)
        return (%res))");

  auto conv_transpose_op_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %packed_weight):
        %res = ipex_prepack::conv_transpose_${op}_run(%input, %packed_weight)
        return (%res))");

  for (auto const& it : utils::supported_unary_post_op_fusion_set()) {
    std::string op = it.first;
    std::string ipex_op_name = it.second.ipex_op_name;

    at::jit::TemplateEnv env;
    env.s("op", op);

    at::jit::TemplateEnv env_fused;
    env_fused.s("op", ipex_op_name);

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        conv_transpose_op_rstring.format(env),
        conv_transpose_op_fused_rstring.format(env_fused));

    auto filters = it.second.filters;
    rewriter.runOnGraph(graph, filters);
  }

  // For non-unary post OPs:
  auto conv_transpose_op_non_unary_rstring = at::jit::CodeTemplate(R"(
     graph(%input, %packed_weight, ${op_input_str}):
        %x : Tensor = ipex_prepack::conv_transpose_run(%input, %packed_weight)
        %res = ${op}(%x, ${op_input_str})
        return (%res))");

  auto conv_transpose_op_non_unary_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %packed_weight, ${op_input_str}):
        %res = ipex_prepack::conv_transpose_${op}_run(%input, ${op_input_str}, %packed_weight)
        return (%res))");

  for (auto const& it : utils::supported_non_unary_post_op_fusion_set()) {
    std::string op = it.first;
    std::string ipex_op_name = it.second.ipex_op_name;
    std::vector<std::string> op_input_list = it.second.op_input_list;
    std::string op_input_str = c10::Join(", ", op_input_list);

    at::jit::TemplateEnv env;
    env.s("op", op);
    env.s("op_input_str", op_input_str);

    at::jit::TemplateEnv env_fused;
    env_fused.s("op", ipex_op_name);
    env_fused.s("op_input_str", op_input_str);

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        conv_transpose_op_non_unary_rstring.format(env),
        conv_transpose_op_non_unary_fused_rstring.format(env_fused));

    auto filters = it.second.filters;
    rewriter.runOnGraph(graph, filters);
  }

  // For conv_transpose-sigmoid-mul
  SubgraphRewriter rewriter_swish;
  std::array<std::string, 2> sigmoid_operators = {"sigmoid", "sigmoid_"};
  std::array<std::string, 2> mul_operators = {"mul", "mul_"};

  auto conv_transpose_sigmoid_mul_rstring = CodeTemplate(R"(
    graph(%input, %packed_weight):
        %x = ipex_prepack::conv_transpose_run(%input, %packed_weight)
        %y = aten::${sigmoid}(%x)
        %res = aten::${mul}(%x, %y)
        return (%res))");

  std::string conv_transpose_swish_fused = R"(
    graph(%input, %packed_weight):
        %res = ipex_prepack::conv_transpose_swish_run(%input, %packed_weight)
        return (%res))";

  for (const auto& sigmoid : sigmoid_operators) {
    TemplateEnv env;
    env.s("sigmoid", sigmoid);
    for (const auto& mul : mul_operators) {
      env.s("mul", mul);
      rewriter_swish.RegisterRewritePattern(
          conv_transpose_sigmoid_mul_rstring.format(env),
          conv_transpose_swish_fused);
    }
  }
  rewriter_swish.runOnGraph(graph);
}

void fuseConvTransposeAdd(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_add_accumu_on_the_right,
      rewriter_add_accumu_on_the_left, rewriter_add_relu;
  std::array<std::string, 2> add_operators = {"add", "add_"};
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};

  // ConvTranspose  accumu
  //       \        /
  //          add
  // output = ConvTranspose + alpha * accumu
  auto conv_transpose_add_accumu_on_the_right_rstring = CodeTemplate(R"(
    graph(%input, %accumu, %alpha, %packed_weight):
        %x = ipex_prepack::conv_transpose_run(%input, %packed_weight)
        %res = aten::${add}(%x, %accumu, %alpha)
        return (%res))");

  //  accumu     ConvTranspose
  //   \        /
  //       add
  // output = accumu + alpha * ConvTranspose, alpha need to be one or none.
  auto conv_transpose_add_accumu_on_the_left_rstring = CodeTemplate(R"(
    graph(%input, %accumu, %alpha, %packed_weight):
        %x = ipex_prepack::conv_transpose_run(%input, %packed_weight)
        %res = aten::${add}(%accumu, %x, %alpha)
        return (%res))");

  std::string conv_transpose_add_fused = R"(
    graph(%input, %accumu, %alpha, %packed_weight):
        %res = ipex_prepack::conv_transpose_add_run(%input, %accumu, %alpha, %packed_weight)
        return (%res))";

  auto conv_transpose_add_relu_rstring = CodeTemplate(R"(
    graph(%input, %accumu, %alpha, %packed_weight):
        %x = ipex_prepack::conv_transpose_add_run(%input, %accumu, %alpha, %packed_weight)
        %res = aten::${relu}(%x)
        return (%res))");

  std::string conv_transpose_add_relu_fused = R"(
    graph(%input, %accumu, %alpha, %packed_weight):
        %res = ipex_prepack::conv_transpose_add_relu_run(%input, %accumu, %alpha, %packed_weight)
        return (%res))";

  // conv_transpose + add
  for (const auto& add : add_operators) {
    TemplateEnv env;
    env.s("add", add);
    rewriter_add_accumu_on_the_right.RegisterRewritePattern(
        conv_transpose_add_accumu_on_the_right_rstring.format(env),
        conv_transpose_add_fused);
    rewriter_add_accumu_on_the_left.RegisterRewritePattern(
        conv_transpose_add_accumu_on_the_left_rstring.format(env),
        conv_transpose_add_fused);
  }

  // conv_transpose + add + relu
  for (const auto& relu : relu_operators) {
    TemplateEnv env;
    env.s("relu", relu);
    rewriter_add_relu.RegisterRewritePattern(
        conv_transpose_add_relu_rstring.format(env),
        conv_transpose_add_relu_fused);
  }

  rewriter_add_accumu_on_the_right.runOnGraph(
      graph, fuse_add_filter_accumu_on_the_right);
  rewriter_add_accumu_on_the_left.runOnGraph(
      graph, fuse_add_filter_accumu_on_the_left);
  rewriter_add_relu.runOnGraph(graph);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
