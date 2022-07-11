#include "csrc/aten/cpu/WeightPack.h"
#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/jit/cpu/kernels/OpContext.h"
#include "csrc/jit/cpu/passes/utils.h"
#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

#include <ATen/code_template.h>

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

using namespace torch_ipex::cpu;
using namespace torch::jit;
using namespace at::jit;

void replaceFrozenIPEXConvWithAtenConv(
    Block* b,
    std::vector<Node*>& get_data_handle_nodes) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      replaceFrozenIPEXConvWithAtenConv(block, get_data_handle_nodes);
    }
    if (n->kind() ==
        Symbol::fromQualString("torch_ipex::convolution_forward")) {
      if (!(constant_as<at::Tensor>(n->namedInput("weight")).has_value())) {
        continue;
      }

      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      auto prepack_node = n->inputs().at(3)->node()->inputs().at(0);
      // For graph before "freeze", cannot get custom class to repack
      if (!toIValue(prepack_node).has_value())
        continue;
      auto conv_op_ctx =
          toIValue(prepack_node).value().toCustomClass<ConvolutionOpContext>();
      at::Tensor weight_tensor = conv_op_ctx->to_public(
          constant_as<at::Tensor>(n->namedInput("weight")).value());
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();

      auto aten_conv = graph->insertNode(graph->create(
          input_size_option.value().size() == 4 ? aten::conv2d : aten::conv3d,
          1));
      aten_conv->addInput(n->inputs().at(0));
      IValue weight_value(weight_tensor);
      auto weight = graph->insertConstant(weight_value);
      aten_conv->addInput(weight);
      aten_conv->addInput(n->inputs().at(2));
      IValue stride_value(conv_op_ctx->get_stride());
      auto stride = graph->insertConstant(stride_value);
      aten_conv->addInput(stride);
      IValue padding_value(conv_op_ctx->get_padding());
      auto padding = graph->insertConstant(padding_value);
      aten_conv->addInput(padding);
      IValue dilation_value(conv_op_ctx->get_dilation());
      auto dilation = graph->insertConstant(dilation_value);
      aten_conv->addInput(dilation);
      IValue groups_value(conv_op_ctx->get_groups());
      auto groups = graph->insertConstant(groups_value);
      aten_conv->addInput(groups);
      aten_conv->output()->setType(n->output()->type()->cast<TensorType>());
      n->output()->replaceAllUsesWith(aten_conv->output());
      get_data_handle_nodes.emplace_back(n->inputs().at(3)->node());
    }
  }
  EliminateDeadCode(b);
}

void replaceFrozenIPEXConvWithAtenConv(std::shared_ptr<Graph>& graph) {
  std::vector<Node*> get_data_handle_nodes;
  replaceFrozenIPEXConvWithAtenConv(graph->block(), get_data_handle_nodes);
  for (auto& n : get_data_handle_nodes) {
    n->destroy();
  }
  EliminateDeadCode(graph);
}

void insertPrePackedConvOp(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      insertPrePackedConvOp(block);
    }
    if (n->kind() == aten::conv1d || n->kind() == aten::conv2d ||
        n->kind() == aten::conv3d) {
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();
      Node* prepack_node;
      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      // if can't get input shape info, will not do weight prepack.
      if (!(input_size_option.has_value() &&
            (input_size_option.value().size() == 3 ||
             input_size_option.value().size() == 4 ||
             input_size_option.value().size() == 5))) {
        continue;
      }
      IValue input_size_value(input_size_option.value());
      if (n->kind() == aten::conv1d || n->kind() == aten::conv2d ||
          n->kind() == aten::conv3d) {
        auto weight_tensor_type = n->inputs().at(1)->type()->cast<TensorType>();
        auto weight_size_option = weight_tensor_type->sizes().concrete_sizes();
        // weight has not shape info, will not do weight prapacked.
        if (!(weight_size_option.has_value() &&
              (weight_size_option.value().size() == 3 ||
               weight_size_option.value().size() == 4 ||
               weight_size_option.value().size() == 5))) {
          continue;
        }
        const auto dtype = weight_tensor_type->scalarType();
        if (dtype.has_value() && *dtype == at::ScalarType::BFloat16 &&
            !ideep::has_bf16_type_support()) {
          continue;
        }
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

        // Note that once creating this "convolution_prepack" node, make sure it
        // is also inserted into the graph. Details ref to "linear_prepack"
        // creation in "graph_rewrite_linear.cpp"
        prepack_node = graph->create(
            Symbol::fromQualString("ipex_prepack::convolution_prepack"), 1);
        for (auto i = 1; i < n->inputs().size() - 1; ++i) {
          Value* v = n->inputs().at(i);
          prepack_node->addInput(v);
        }
        // add conv groups
        prepack_node->addInput(n->inputs().at(n->inputs().size() - 1));
        prepack_node->addInput(weight_is_channels_last);
      } else {
        prepack_node = graph->create(
            Symbol::fromQualString("ipex_prepack::convolution_prepack"), 1);
        for (auto i = 1; i < n->inputs().size(); ++i) {
          Value* v = n->inputs().at(i);
          prepack_node->addInput(v);
        }
      }
      auto input_size = graph->insertConstant(input_size_value);
      prepack_node->addInput(input_size);
      prepack_node->output()->setType(getCustomClass(
          "__torch__.torch.classes.ipex_prepack.ConvolutionOpContext"));

      graph->insertNode(prepack_node);
      auto prepack_conv = graph->insertNode(graph->create(
          Symbol::fromQualString("ipex_prepack::convolution_run"), 1));
      prepack_conv->addInput(n->inputs().at(0));
      prepack_conv->addInput(prepack_node->output());
      prepack_conv->output()->setType(n->output()->type()->cast<TensorType>());
      auto v = n->outputs().at(0);
      n->output()->replaceAllUsesWith(prepack_conv->output());
    }
  }
  EliminateDeadCode(b);
}

void insertPrePackedConvOp(std::shared_ptr<Graph>& graph) {
  insertPrePackedConvOp(graph->block());
}

void fuseConvWithEltwise(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_swish;
  std::array<std::string, 2> sigmoid_operators = {"sigmoid", "sigmoid_"};
  std::array<std::string, 2> mul_operators = {"mul", "mul_"};

  // For unary post OPs:
  auto conv_op_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size)
        %x : Tensor = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = ${op}(%x)
        return (%res))");

  auto conv_op_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_${op}_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size)
        %res = ipex_prepack::convolution_${op}_run(%input, %packed_weight)
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
        conv_op_rstring.format(env), conv_op_fused_rstring.format(env_fused));

    auto filters = it.second.filters;
    rewriter.runOnGraph(graph, filters);
  }

  // For non-unary post OPs:
  auto conv_op_non_unary_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[], ${op_input_str}):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size)
        %x : Tensor = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = ${op}(%x, ${op_input_str})
        return (%res))");

  auto conv_op_non_unary_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[], ${op_input_str}):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_${op}_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size, ${op_input_str})
        %res = ipex_prepack::convolution_${op}_run(%input, ${op_input_str}, %packed_weight)
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
        conv_op_non_unary_rstring.format(env),
        conv_op_non_unary_fused_rstring.format(env_fused));

    auto filters = it.second.filters;
    rewriter.runOnGraph(graph, filters);
  }

  auto conv_sigmoid_mul_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %y = aten::${sigmoid}(%x)
        %res = aten::${mul}(%x, %y)
        return (%res))");

  std::string conv_swish_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_swish_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size)
        %res = ipex_prepack::convolution_swish_run(%input, %packed_weight)
        return (%res))";

  for (const auto& sigmoid : sigmoid_operators) {
    TemplateEnv env;
    env.s("sigmoid", sigmoid);
    for (const auto& mul : mul_operators) {
      env.s("mul", mul);
      rewriter_swish.RegisterRewritePattern(
          conv_sigmoid_mul_rstring.format(env), conv_swish_fused);
    }
  }

  rewriter_swish.runOnGraph(graph);
}

void fuseConvAddRelu(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_add_accumu_on_the_right,
      rewriter_add_accumu_on_the_left, rewriter_add_relu;
  std::array<std::string, 2> add_operators = {"add", "add_"};
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};

  // conv   Y
  //   \   /
  //    add
  // output = conv_output + alpha*Y
  auto conv_add_accumu_on_the_right_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last, %input_size:int[]):
        %packed_weight = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${add}(%x, %accumu, %alpha) return (%res))");

  //  Y     conv
  //   \   /
  //    add
  // output = Y + alpha*conv_output, alpha need to one or none.
  auto conv_add_accumu_on_the_left_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %groups,  %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${add}(%accumu, %x, %alpha) return (%res))");

  std::string conv_add_fused = R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_add_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size, %alpha)
        %res = ipex_prepack::convolution_add_run(%input, %accumu, %alpha, %packed_weight)
        return (%res))";

  auto conv_add_relu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_add_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size, %alpha)
        %x = ipex_prepack::convolution_add_run(%input, %accumu, %alpha, %packed_weight)
        %res = aten::${relu}(%x) return (%res))");

  std::string conv_add_relu_fused = R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_add_relu_prepack(%weight, %bias, %stride, %padding, %dilation, %groups, %weight_is_channels_last, %input_size, %alpha)
        %res = ipex_prepack::convolution_add_relu_run(%input, %accumu, %alpha, %packed_weight) return (%res))";

  // conv+add
  for (const auto& add : add_operators) {
    TemplateEnv env;
    env.s("add", add);
    rewriter_add_accumu_on_the_right.RegisterRewritePattern(
        conv_add_accumu_on_the_right_rstring.format(env), conv_add_fused);
    rewriter_add_accumu_on_the_left.RegisterRewritePattern(
        conv_add_accumu_on_the_left_rstring.format(env), conv_add_fused);
  }

  // fused_conv_add+relu
  for (const auto& relu : relu_operators) {
    TemplateEnv env;
    env.s("relu", relu);
    rewriter_add_relu.RegisterRewritePattern(
        conv_add_relu_rstring.format(env), conv_add_relu_fused);
  }

  rewriter_add_accumu_on_the_right.runOnGraph(
      graph, fuse_add_filter_accumu_on_the_right);
  rewriter_add_accumu_on_the_left.runOnGraph(
      graph, fuse_add_filter_accumu_on_the_left);
  rewriter_add_relu.runOnGraph(graph);
}

void fuseBottleneck(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_v1, rewriter_v2;
  std::string bottleneck_v1 = R"(
    graph(%input, %packed_weight1, %packed_weight2, %packed_weight3, %alpha):
        %res1 = ipex_prepack::convolution_relu_run(%input, %packed_weight1)
        %res2 = ipex_prepack::convolution_relu_run(%res1, %packed_weight2)
        %res = ipex_prepack::convolution_add_relu_run(%res2, %input, %alpha, %packed_weight3)
        return (%res))";
  std::string bottleneck_fused_v1 = R"(
    graph(%input, %packed_weight1, %packed_weight2, %packed_weight3, %alpha):
        %res = ipex_prepack::convolution_bottleneck_run(%input, %packed_weight1, %packed_weight2, %packed_weight3)
        return (%res))";

  std::string bottleneck_v2 = R"(
    graph(%input, %packed_weight1, %packed_weight2, %packed_weight3, %packed_weight4, %alpha):
        %res1 = ipex_prepack::convolution_relu_run(%input, %packed_weight1)
        %res2 = ipex_prepack::convolution_relu_run(%res1, %packed_weight2)
        %res3 = ipex_prepack::convolution_run(%input, %packed_weight3)
        %res = ipex_prepack::convolution_add_relu_run(%res2, %res3, %alpha, %packed_weight4)
        return (%res))";
  std::string bottleneck_fused_v2 = R"(
    graph(%input, %packed_weight1, %packed_weight2, %packed_weight3, %packed_weight4, %alpha):
        %res = ipex_prepack::convolution_bottleneck_run(%input, %packed_weight1, %packed_weight2, %packed_weight3, %packed_weight4)
        return (%res))";

  // Requires weights are prepacked and expect channels last activation, biases
  // exist and alpha is constant. For this case, there will support a fast path
  // which has't check in convolution ops(such as format check and desc check)
  // and format reorder, which can reduce many integration overhead in FW dide.
  auto filter_v1 = [](const Match& match,
                      const std::unordered_map<std::string, Value*>& vmap) {
    auto packed_weight1 =
        match.values_map.at(vmap.at("packed_weight1"))->node();
    auto packed_weight2 =
        match.values_map.at(vmap.at("packed_weight2"))->node();
    auto packed_weight3 =
        match.values_map.at(vmap.at("packed_weight3"))->node();

    auto weight1_is_channels_last =
        constant_as<bool>(packed_weight1->inputs().at(6)).value();
    auto weight2_is_channels_last =
        constant_as<bool>(packed_weight2->inputs().at(6)).value();
    auto weight3_is_channels_last =
        constant_as<bool>(packed_weight3->inputs().at(6)).value();
    if (!weight1_is_channels_last || !weight2_is_channels_last ||
        !weight3_is_channels_last) {
      return false;
    }

    auto bias1_type = packed_weight1->inputs().at(1)->type();
    auto bias2_type = packed_weight2->inputs().at(1)->type();
    auto bias3_type = packed_weight3->inputs().at(1)->type();
    if (bias1_type == NoneType::get() || bias2_type == NoneType::get() ||
        bias3_type == NoneType::get()) {
      return false;
    }

    auto alpha = match.values_map.at(vmap.at("alpha"))->node();
    if (alpha->kind() != prim::Constant) {
      return false;
    }
    return true;
  };

  auto filter_v2 = [](const Match& match,
                      const std::unordered_map<std::string, Value*>& vmap) {
    auto packed_weight1 =
        match.values_map.at(vmap.at("packed_weight1"))->node();
    auto packed_weight2 =
        match.values_map.at(vmap.at("packed_weight2"))->node();
    auto packed_weight3 =
        match.values_map.at(vmap.at("packed_weight3"))->node();
    auto packed_weight4 =
        match.values_map.at(vmap.at("packed_weight4"))->node();

    auto weight1_is_channels_last =
        constant_as<bool>(packed_weight1->inputs().at(6)).value();
    auto weight2_is_channels_last =
        constant_as<bool>(packed_weight2->inputs().at(6)).value();
    auto weight3_is_channels_last =
        constant_as<bool>(packed_weight3->inputs().at(6)).value();
    auto weight4_is_channels_last =
        constant_as<bool>(packed_weight4->inputs().at(6)).value();
    if (!weight1_is_channels_last || !weight2_is_channels_last ||
        !weight3_is_channels_last || !weight4_is_channels_last) {
      return false;
    }

    auto bias1_type = packed_weight1->inputs().at(1)->type();
    auto bias2_type = packed_weight2->inputs().at(1)->type();
    auto bias3_type = packed_weight3->inputs().at(1)->type();
    auto bias4_type = packed_weight3->inputs().at(1)->type();
    if (bias1_type == NoneType::get() || bias2_type == NoneType::get() ||
        bias3_type == NoneType::get() || bias4_type == NoneType::get()) {
      return false;
    }

    auto alpha = match.values_map.at(vmap.at("alpha"))->node();
    if (alpha->kind() != prim::Constant) {
      return false;
    }
    return true;
  };

  rewriter_v1.RegisterRewritePattern(bottleneck_v1, bottleneck_fused_v1);
  rewriter_v2.RegisterRewritePattern(bottleneck_v2, bottleneck_fused_v2);
  rewriter_v1.runOnGraph(graph, filter_v1);
  rewriter_v2.runOnGraph(graph, filter_v2);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
