#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

#include <ATen/code_template.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;

void insertPrePackedLinearOp(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      insertPrePackedLinearOp(block);
    }
    if (n->kind() == aten::linear ||
        n->kind() == Symbol::fromQualString("torch_ipex::ipex_linear")) {
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();
      auto prepack_node = graph->create(
          Symbol::fromQualString("ipex_prepack::linear_prepack"), 1);
      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      if (!(input_size_option.has_value() &&
            input_size_option.value().size() >= 2)) {
        continue;
      }
      auto input_size = input_size_option.value();
      int64_t b_size = std::accumulate(
                           input_size.begin(),
                           input_size.end(),
                           1,
                           std::multiplies<double>()) /
          input_size[input_size.size() - 1];
      IValue batch_size_value(b_size);
      auto batch_size = graph->insertConstant(batch_size_value);
      if (n->kind() == aten::linear) {
        auto tt = n->inputs().at(1)->type()->cast<TensorType>();
        auto weight_size_option = tt->sizes().concrete_sizes();
        if (!(weight_size_option.has_value() &&
              weight_size_option.value().size() == 2)) {
          continue;
        }
        auto weight_dtype_option = tt->scalarType();
        if (!(weight_dtype_option.has_value() &&
              weight_dtype_option.value() == at::ScalarType::BFloat16)) {
          continue;
        }
        auto weight_size = weight_size_option.value();
        int64_t o_channel = weight_size[0];
        int64_t i_channel = weight_size[1];
        IValue weight_is_prepacked_value(false),
            output_channel_value(o_channel), input_channel_value(i_channel);
        auto output_channel = graph->insertConstant(output_channel_value);
        auto input_channel = graph->insertConstant(input_channel_value);
        auto weight_is_prepacked =
            graph->insertConstant(weight_is_prepacked_value);
        for (auto i = 1; i < n->inputs().size(); ++i) {
          Value* v = n->inputs().at(i);
          prepack_node->addInput(v);
        }
        prepack_node->addInput(output_channel);
        prepack_node->addInput(input_channel);
        prepack_node->addInput(batch_size);
        prepack_node->addInput(weight_is_prepacked);
      } else {
        prepack_node->addInput(n->inputs().at(1));
        prepack_node->addInput(n->inputs().at(4));
        prepack_node->addInput(n->inputs().at(2));
        prepack_node->addInput(n->inputs().at(3));
        prepack_node->addInput(batch_size);
        IValue weight_is_prepacked_value(true);
        auto weight_is_prepacked =
            graph->insertConstant(weight_is_prepacked_value);
        prepack_node->addInput(weight_is_prepacked);
      }
      prepack_node->output()->setType(getCustomClass(
          "__torch__.torch.classes.ipex_prepack.LinearOpContext"));
      graph->insertNode(prepack_node);
      auto prepack_linear = graph->insertNode(
          graph->create(Symbol::fromQualString("ipex_prepack::linear_run"), 1));
      prepack_linear->addInput(n->inputs().at(0));
      prepack_linear->addInput(prepack_node->output());
      prepack_linear->output()->setType(
          n->output()->type()->cast<TensorType>());
      auto v = n->outputs().at(0);
      n->output()->replaceAllUsesWith(prepack_linear->output());
    }
  }
  EliminateDeadCode(b);
}

void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  insertPrePackedLinearOp(graph->block());
}

void fuseLinearWithEltwise(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_relu, rewriter_gelu, rewriter_silu,
      rewriter_sigmoid, rewriter_swish, rewriter_tanh;
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};
  std::array<std::string, 2> sigmoid_operators = {"sigmoid", "sigmoid_"};
  std::array<std::string, 2> silu_operators = {"silu", "silu_"};
  std::array<std::string, 2> mul_operators = {"mul", "mul_"};
  std::array<std::string, 2> tanh_operators = {"tanh", "tanh_"};

  auto linear_relu_rstring = CodeTemplate(R"(
     graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res = aten::${relu}(%x)
        return (%res))");

  std::string linear_relu_fused = R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %res = ipex_prepack::linear_relu_run(%input, %packed_weight)
        return (%res))";

  auto linear_tanh_rstring = CodeTemplate(R"(
     graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res = aten::${tanh}(%x)
        return (%res))");

  std::string linear_tanh_fused = R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %res = ipex_prepack::linear_tanh_run(%input, %packed_weight)
        return (%res))";

  std::string linear_gelu = R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res= aten::gelu(%x)
        return (%res))";

  std::string linear_gelu_fused = R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %res = ipex_prepack::linear_gelu_run(%input, %packed_weight)
        return (%res))";

  auto linear_sigmoid_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res= aten::${sigmoid}(%x)
        return (%res))");

  auto linear_silu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res= aten::${silu}(%x)
        return (%res))");

  auto linear_sigmoid_mul_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %y = aten::${sigmoid}(%x)
        %res = aten::${mul}(%x, %y)
        return (%res))");

  std::string linear_swish_fused = R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %res = ipex_prepack::linear_swish_run(%input, %packed_weight)
        return (%res))";

  std::string linear_sigmoid_fused = R"(
    graph(%input, %weight, %bias, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %res = ipex_prepack::linear_sigmoid_run(%input, %packed_weight)
        return (%res))";

  for (const auto& relu : relu_operators) {
    TemplateEnv env;
    env.s("relu", relu);
    rewriter_relu.RegisterRewritePattern(
        linear_relu_rstring.format(env), linear_relu_fused);
  }

  for (const auto& tanh : tanh_operators) {
    TemplateEnv env;
    env.s("tanh", tanh);
    rewriter_tanh.RegisterRewritePattern(
        linear_tanh_rstring.format(env), linear_tanh_fused);
  }

  for (const auto& silu : silu_operators) {
    TemplateEnv env;
    env.s("silu", silu);
    rewriter_silu.RegisterRewritePattern(
        linear_silu_rstring.format(env), linear_swish_fused);
  }

  for (const auto& sigmoid : sigmoid_operators) {
    TemplateEnv env;
    env.s("sigmoid", sigmoid);
    rewriter_sigmoid.RegisterRewritePattern(
        linear_sigmoid_rstring.format(env), linear_sigmoid_fused);
    for (const auto& mul : mul_operators) {
      env.s("mul", mul);
      rewriter_swish.RegisterRewritePattern(
          linear_sigmoid_mul_rstring.format(env), linear_swish_fused);
    }
  }
  rewriter_silu.runOnGraph(graph);
  rewriter_sigmoid.runOnGraph(graph);
  rewriter_swish.runOnGraph(graph);
  rewriter_gelu.RegisterRewritePattern(linear_gelu, linear_gelu_fused);

  rewriter_relu.runOnGraph(graph);
  rewriter_tanh.runOnGraph(graph);
  rewriter_gelu.runOnGraph(graph);
}

void fuseLinearAddRelu(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_add_v1, rewriter_add_v2;
  std::array<std::string, 2> add_operators = {"add", "add_"};

  // linear   Y
  //   \   /
  //    add
  // output = linear_output + alpha*Y
  auto linear_add_rstring_v1 = CodeTemplate(R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res = aten::${add}(%x, %accumu, %alpha)
        return (%res))");

  //  Y     linear
  //   \   /
  //    add
  // output = Y + alpha*linear_output, alpha need to one or none.
  auto linear_add_rstring_v2 = CodeTemplate(R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %x = ipex_prepack::linear_run(%input, %packed_weight)
        %res = aten::${add}(%accumu, %x, %alpha)
        return (%res))");

  std::string linear_add_fused = R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %out_features:int, %in_features:int, %batch_size:int, %weight_is_prepacked:bool):
        %packed_weight = ipex_prepack::linear_prepack(%weight, %bias, %out_features, %in_features, %batch_size, %weight_is_prepacked)
        %res = ipex_prepack::linear_add_run(%input, %accumu, %alpha, %packed_weight)
        return (%res))";

  // linear + add
  for (const auto& add : add_operators) {
    TemplateEnv env;
    env.s("add", add);
    rewriter_add_v1.RegisterRewritePattern(
        linear_add_rstring_v1.format(env), linear_add_fused);
    rewriter_add_v2.RegisterRewritePattern(
        linear_add_rstring_v2.format(env), linear_add_fused);
  }

  rewriter_add_v1.runOnGraph(graph, fuse_add_filter_v1);
  rewriter_add_v2.runOnGraph(graph, fuse_add_filter_v2);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
