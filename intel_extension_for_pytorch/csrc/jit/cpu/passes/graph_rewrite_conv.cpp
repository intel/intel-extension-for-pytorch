#include "csrc/aten/cpu/WeightPack.h"
#include "csrc/jit/cpu/kernels/OpContext.h"
#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

#include <ATen/code_template.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;
using namespace torch_ipex::cpu;

void replaceFrozenIPEXConvWithAtenConv(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      replaceFrozenIPEXConvWithAtenConv(block);
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
      // For graph before "freeze", cannot get custom class to repack
      auto prepack_node = n->inputs().at(3);
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
    }
  }
  EliminateDeadCode(b);
}

void replaceFrozenIPEXConvWithAtenConv(std::shared_ptr<Graph>& graph) {
  replaceFrozenIPEXConvWithAtenConv(graph->block());
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
      auto prepack_node = graph->create(
          Symbol::fromQualString("ipex_prepack::convolution_prepack"), 1);
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
        auto weight_size_option = n->inputs()
                                      .at(1)
                                      ->type()
                                      ->cast<TensorType>()
                                      ->sizes()
                                      .concrete_sizes();
        // weight has not shape info, will not do weight prapacked.
        if (!(weight_size_option.has_value() &&
              (weight_size_option.value().size() == 3 ||
               weight_size_option.value().size() == 4 ||
               weight_size_option.value().size() == 5))) {
          continue;
        }
        auto weight_size = weight_size_option.value();
        // 1d case.
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
        bool w_is_channels_last = false;
        if (constant_as<at::Tensor>(n->namedInput("weight")).has_value()) {
          at::Tensor weight_tensor =
              constant_as<at::Tensor>(n->namedInput("weight")).value();
          w_is_channels_last =
              weight_tensor.is_contiguous(at::MemoryFormat::ChannelsLast) ||
              weight_tensor.is_contiguous(at::MemoryFormat::ChannelsLast3d);
        }
        int64_t o_channel = weight_size[0];
        IValue kernel_size_value(k_size),
            weight_is_channels_last_value(w_is_channels_last),
            output_channel_value(o_channel);

        auto kernel_size = graph->insertConstant(kernel_size_value);
        auto weight_is_channels_last =
            graph->insertConstant(weight_is_channels_last_value);
        auto output_channel = graph->insertConstant(output_channel_value);

        for (auto i = 1; i < n->inputs().size() - 1; ++i) {
          Value* v = n->inputs().at(i);
          prepack_node->addInput(v);
        }
        prepack_node->addInput(kernel_size);
        // add conv groups
        prepack_node->addInput(n->inputs().at(n->inputs().size() - 1));
        prepack_node->addInput(output_channel);
        prepack_node->addInput(weight_is_channels_last);
      } else {
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
  SubgraphRewriter rewriter_relu, rewriter_sigmoid, rewriter_hardtanh,
      rewriter_elu, rewriter_swish, rewriter_silu, rewriter_leaky_relu,
      rewriter_gelu;
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};
  std::array<std::string, 2> sigmoid_operators = {"sigmoid", "sigmoid_"};
  std::array<std::string, 2> hardtanh_operators = {"hardtanh", "hardtanh_"};
  std::array<std::string, 2> elu_operators = {"elu", "elu_"};
  std::array<std::string, 2> mul_operators = {"mul", "mul_"};
  std::array<std::string, 2> silu_operators = {"silu", "silu_"};
  std::array<std::string, 2> leaky_relu_operators = {
      "leaky_relu", "leaky_relu_"};

  auto conv_relu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${relu}(%x)
        return (%res))");

  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_relu_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %res = ipex_prepack::convolution_relu_run(%input, %packed_weight)
        return (%res))";

  auto conv_sigmoid_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${sigmoid}(%x)
        return (%res))");

  std::string conv_sigmoid_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_sigmoid_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %res = ipex_prepack::convolution_sigmoid_run(%input, %packed_weight)
        return (%res))";

  auto conv_hardtanh_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[], %min, %max):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${hardtanh}(%x, %min, %max)
        return (%res))");

  std::string conv_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[], %min, %max):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_hardtanh_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size, %min, %max)
        %res = ipex_prepack::convolution_hardtanh_run(%input, %min, %max, %packed_weight)
        return (%res))";

  auto conv_elu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[], %alpha, %scale, %input_scale):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel,  %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${elu}(%x, %alpha, %scale, %input_scale)
        return (%res))");

  std::string conv_elu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[], %alpha, %scale, %input_scale):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_elu_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel,  %weight_is_channels_last, %input_size, %alpha, %scale, %input_scale)
        %res = ipex_prepack::convolution_elu_run(%input, %alpha, %scale, %input_scale, %packed_weight)
        return (%res))";

  auto conv_sigmoid_mul_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %y = aten::${sigmoid}(%x)
        %res = aten::${mul}(%x, %y)
        return (%res))");

  auto conv_silu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight: __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${silu}(%x)
        return (%res))");

  std::string conv_swish_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_swish_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %res = ipex_prepack::convolution_swish_run(%input, %packed_weight)
        return (%res))";

  auto conv_leaky_relu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[], %alpha):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${leaky_relu}(%x, %alpha)
        return (%res))");

  std::string conv_leaky_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[], %alpha):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_leaky_relu_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size, %alpha)
        %res = ipex_prepack::convolution_leaky_relu_run(%input, %alpha, %packed_weight)
        return (%res))";

  auto conv_gelu_rstring = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[], %approximate):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::gelu(%x, %approximate)
        return (%res))";

  std::string conv_gelu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[], %approximate):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_gelu_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size, %approximate)
        %res = ipex_prepack::convolution_gelu_run(%input, %approximate, %packed_weight)
        return (%res))";

  for (const auto& relu : relu_operators) {
    TemplateEnv env;
    env.s("relu", relu);
    rewriter_relu.RegisterRewritePattern(
        conv_relu_rstring.format(env), conv_relu_fused);
  }

  for (const auto& sigmoid : sigmoid_operators) {
    TemplateEnv env;
    env.s("sigmoid", sigmoid);
    rewriter_sigmoid.RegisterRewritePattern(
        conv_sigmoid_rstring.format(env), conv_sigmoid_fused);
    for (const auto& mul : mul_operators) {
      env.s("mul", mul);
      rewriter_swish.RegisterRewritePattern(
          conv_sigmoid_mul_rstring.format(env), conv_swish_fused);
    }
  }
  for (const auto& silu : silu_operators) {
    TemplateEnv env;
    env.s("silu", silu);
    rewriter_silu.RegisterRewritePattern(
        conv_silu_rstring.format(env), conv_swish_fused);
  }

  for (const auto& hardtanh : hardtanh_operators) {
    TemplateEnv env;
    env.s("hardtanh", hardtanh);
    rewriter_hardtanh.RegisterRewritePattern(
        conv_hardtanh_rstring.format(env), conv_hardtanh_fused);
  }

  for (const auto& elu : elu_operators) {
    TemplateEnv env;
    env.s("elu", elu);
    rewriter_elu.RegisterRewritePattern(
        conv_elu_rstring.format(env), conv_elu_fused);
  }

  auto filter_conv2d_elu =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        const auto& match_vmap = match.values_map;
        auto input_scale_value =
            getIValue("input_scale", match_vmap, vmap).value();
        bool no_input_scale = input_scale_value.isDouble()
            ? (input_scale_value.toDouble() == 1.0)
            : (input_scale_value.toInt() == 1);
        return no_input_scale;
      };

  for (const auto& leaky_relu : leaky_relu_operators) {
    TemplateEnv env;
    env.s("leaky_relu", leaky_relu);
    rewriter_leaky_relu.RegisterRewritePattern(
        conv_leaky_relu_rstring.format(env), conv_leaky_relu_fused);
  }

  auto filter_conv_gelu =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        const auto& match_vmap = match.values_map;
        auto approximate_value =
            getIValue("approximate", match_vmap, vmap).value();
        return approximate_value == "none" || approximate_value == "tanh";
      };

  rewriter_gelu.RegisterRewritePattern(conv_gelu_rstring, conv_gelu_fused);

  rewriter_relu.runOnGraph(graph);
  rewriter_sigmoid.runOnGraph(graph);
  rewriter_hardtanh.runOnGraph(graph);
  rewriter_elu.runOnGraph(graph, filter_conv2d_elu);
  rewriter_swish.runOnGraph(graph);
  rewriter_silu.runOnGraph(graph);
  rewriter_leaky_relu.runOnGraph(graph);
  rewriter_gelu.runOnGraph(graph, filter_conv_gelu);
}

void fuseConvAddRelu(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_add_v1, rewriter_add_v2, rewriter_add_relu;
  std::array<std::string, 2> add_operators = {"add", "add_"};
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};

  // conv   Y
  //   \   /
  //    add
  // output = conv_output + alpha*Y
  auto conv_add_rstring_v1 = CodeTemplate(R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last, %input_size:int[]):
        %packed_weight = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${add}(%x, %accumu, %alpha) return (%res))");

  //  Y     conv
  //   \   /
  //    add
  // output = Y + alpha*conv_output, alpha need to one or none.
  auto conv_add_rstring_v2 = CodeTemplate(R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight = ipex_prepack::convolution_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel,  %weight_is_channels_last, %input_size)
        %x = ipex_prepack::convolution_run(%input, %packed_weight)
        %res = aten::${add}(%accumu, %x, %alpha) return (%res))");

  std::string conv_add_fused = R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_add_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size, %alpha)
        %res = ipex_prepack::convolution_add_run(%input, %accumu, %alpha, %packed_weight)
        return (%res))";

  auto conv_add_relu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_add_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size, %alpha)
        %x = ipex_prepack::convolution_add_run(%input, %accumu, %alpha, %packed_weight)
        %res = aten::${relu}(%x) return (%res))");

  std::string conv_add_relu_fused = R"(
    graph(%input, %weight, %bias, %accumu, %alpha, %stride:int[], %padding:int[], %dilation:int[], %kernel_size:int[], %groups:int, %output_channel:int, %weight_is_channels_last:bool, %input_size:int[]):
        %packed_weight : __torch__.torch.classes.ipex_prepack.ConvolutionOpContext = ipex_prepack::convolution_add_relu_prepack(%weight, %bias, %stride, %padding, %dilation, %kernel_size, %groups, %output_channel, %weight_is_channels_last, %input_size, %alpha)
        %res = ipex_prepack::convolution_add_relu_run(%input, %accumu, %alpha, %packed_weight) return (%res))";

  // conv+add
  for (const auto& add : add_operators) {
    TemplateEnv env;
    env.s("add", add);
    rewriter_add_v1.RegisterRewritePattern(
        conv_add_rstring_v1.format(env), conv_add_fused);
    rewriter_add_v2.RegisterRewritePattern(
        conv_add_rstring_v2.format(env), conv_add_fused);
  }

  // fused_conv_add+relu
  for (const auto& relu : relu_operators) {
    TemplateEnv env;
    env.s("relu", relu);
    rewriter_add_relu.RegisterRewritePattern(
        conv_add_relu_rstring.format(env), conv_add_relu_fused);
  }

  rewriter_add_v1.runOnGraph(graph, fuse_add_filter_v1);
  rewriter_add_v2.runOnGraph(graph, fuse_add_filter_v2);
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
        constant_as<bool>(packed_weight1->inputs().at(8)).value();
    auto weight2_is_channels_last =
        constant_as<bool>(packed_weight2->inputs().at(8)).value();
    auto weight3_is_channels_last =
        constant_as<bool>(packed_weight3->inputs().at(8)).value();
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
        constant_as<bool>(packed_weight1->inputs().at(8)).value();
    auto weight2_is_channels_last =
        constant_as<bool>(packed_weight2->inputs().at(8)).value();
    auto weight3_is_channels_last =
        constant_as<bool>(packed_weight3->inputs().at(8)).value();
    auto weight4_is_channels_last =
        constant_as<bool>(packed_weight4->inputs().at(8)).value();
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
} // namespace torch
