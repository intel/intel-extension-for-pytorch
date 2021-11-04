
#include "graph_rewrite.h"

namespace torch {
namespace jit {
namespace graph_rewrite {

// those code just copy from PyTorch offical:
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/graph_rewrite_helper.cpp

Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  return match_vmap.at(vmap.at(name));
}

c10::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  return toIValue(getValue(name, match_vmap, vmap));
}

std::unordered_map<std::string, c10::IValue> getConvParams(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  std::unordered_map<std::string, c10::IValue> calc_values;
  const auto& match_vmap = match.values_map;
  auto transposed_value = getIValue("transposed", match_vmap, vmap).value();
  calc_values["transposed"] = transposed_value;
  auto benchmark_value = getIValue("benchmark", match_vmap, vmap).value();
  calc_values["benchmark"] = benchmark_value;
  auto deterministic_value =
      getIValue("deterministic", match_vmap, vmap).value();
  calc_values["deterministic"] = deterministic_value;
  auto cudnn_enabled_value =
      getIValue("cudnn_enabled", match_vmap, vmap).value();
  calc_values["cudnn_enabled"] = cudnn_enabled_value;
  auto output_padding_value =
      getIValue("output_padding", match_vmap, vmap).value();
  calc_values["output_padding"] = output_padding_value;
  auto stride_value = getIValue("stride", match_vmap, vmap).value();
  calc_values["stride"] = stride_value;
  auto padding_value = getIValue("padding", match_vmap, vmap).value();
  calc_values["padding"] = padding_value;
  auto dilation_value = getIValue("dilation", match_vmap, vmap).value();
  calc_values["dilation"] = dilation_value;
  auto allow_tf32_value = getIValue("allow_tf32", match_vmap, vmap).value();
  calc_values["allow_tf32_value"] = allow_tf32_value;
  return calc_values;
}

void replaceConvolutionWithAtenConv(std::shared_ptr<Graph>& graph) {
  // TODO: remove constant prop in the pass
  ConstantPropagation(graph);
  std::string convolution_deprecated = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled)
        return (%r) )";

  std::string convolution = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled, %allow_tf32)
        return (%r) )";

  std::string conv2d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
  std::string conv2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv1d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv1d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
  std::string conv1d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv1d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv3d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv3d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
  std::string conv3d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv3d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv_transpose1d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv_transpose1d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose2d_for_deprecated_conv = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv_transpose2d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv_transpose2d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  // Filter the unsupported case
  auto filter_conv1d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 1 ||
        calc_value_map["stride"].toIntList().size() != 1 ||
        calc_value_map["padding"].toIntList().size() != 1 ||
        calc_value_map["dilation"].toIntList().size() != 1) {
      return false;
    }
    return !calc_value_map["transposed"].toBool();
  };
  auto filter_conv2d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 2 ||
        calc_value_map["stride"].toIntList().size() != 2 ||
        calc_value_map["padding"].toIntList().size() != 2 ||
        calc_value_map["dilation"].toIntList().size() != 2) {
      return false;
    }
    return !calc_value_map["transposed"].toBool();
  };
  auto filter_conv3d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 3 ||
        calc_value_map["stride"].toIntList().size() != 3 ||
        calc_value_map["padding"].toIntList().size() != 3 ||
        calc_value_map["dilation"].toIntList().size() != 3) {
      return false;
    }
    return !calc_value_map["transposed"].toBool();
  };
  auto filter_conv_transpose1d =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        auto calc_value_map = getConvParams(match, vmap);
        if (calc_value_map["output_padding"].toIntList().size() != 1 ||
            calc_value_map["stride"].toIntList().size() != 1 ||
            calc_value_map["padding"].toIntList().size() != 1 ||
            calc_value_map["dilation"].toIntList().size() != 1) {
          return false;
        }
        return calc_value_map["transposed"].toBool();
      };
  auto filter_conv_transpose2d =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        auto calc_value_map = getConvParams(match, vmap);
        if (calc_value_map["output_padding"].toIntList().size() != 2 ||
            calc_value_map["stride"].toIntList().size() != 2 ||
            calc_value_map["padding"].toIntList().size() != 2 ||
            calc_value_map["dilation"].toIntList().size() != 2) {
          return false;
        }
        return calc_value_map["transposed"].toBool();
      };

  IpexSubgraphRewriter rewriter_conv1d;
  rewriter_conv1d.RegisterRewritePattern(convolution, conv1d);
  rewriter_conv1d.RegisterRewritePattern(
      convolution_deprecated, conv1d_for_deprecated_conv);
  rewriter_conv1d.runOnGraph(graph, filter_conv1d);

  IpexSubgraphRewriter rewriter_conv2d;
  rewriter_conv2d.RegisterRewritePattern(convolution, conv2d);
  rewriter_conv2d.RegisterRewritePattern(
      convolution_deprecated, conv2d_for_deprecated_conv);
  rewriter_conv2d.runOnGraph(graph, filter_conv2d);

  IpexSubgraphRewriter rewriter_conv3d;
  rewriter_conv3d.RegisterRewritePattern(convolution, conv3d);
  rewriter_conv3d.RegisterRewritePattern(
      convolution_deprecated, conv3d_for_deprecated_conv);
  rewriter_conv3d.runOnGraph(graph, filter_conv3d);

  IpexSubgraphRewriter rewriter_conv_transpose1d;
  rewriter_conv_transpose1d.RegisterRewritePattern(
      convolution, conv_transpose1d);
  rewriter_conv_transpose1d.runOnGraph(graph, filter_conv_transpose1d);

  IpexSubgraphRewriter rewriter_conv_transpose2d;
  rewriter_conv_transpose2d.RegisterRewritePattern(
      convolution, conv_transpose2d);
  rewriter_conv_transpose2d.RegisterRewritePattern(
      convolution_deprecated, conv_transpose2d_for_deprecated_conv);
  rewriter_conv_transpose2d.runOnGraph(graph, filter_conv_transpose2d);
}

void FuseShuffle(std::shared_ptr<Graph>& graph) {
  std::string shuffle = R"(
      graph(%input, %view_shape:int[], %trans_dim0:int, %trans_dim1:int, %mem_format:int, %flattern_shape:int[]):
        %r1 = aten::view(%input, %view_shape)
        %r2 = aten::transpose(%r1, %trans_dim0, %trans_dim1)
        %r3 = aten::contiguous(%r2, %mem_format)
        %r4 = aten::view(%r3, %flattern_shape)
        return (%r4) )";

  std::string shuffle_2d_fusion = R"(
      graph(%input, %view_shape:int[], %trans_dim0:int, %trans_dim1:int, %mem_format:int, %flattern_shape:int[]):
        %r = ipex::shuffle_2d(%input, %view_shape, %trans_dim0, %trans_dim1)
        return (%r) )";

  auto filter_shuffle_2d_fusion = [] (
      const Match& match,
      const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto input_ = getIValue("input", match_vmap, vmap).value();
    if (!(input_.isTensor())) {
      return false;
    }
    auto view_shape_ = getIValue("view_shape", match_vmap, vmap).value();
    if (!(view_shape_.isIntList())) {
      return false;
    }
    auto trans_dim0_ = getIValue("trans_dim0", match_vmap, vmap).value();
    if (!(trans_dim0_.isInt())) {
      return false;
    }
    auto trans_dim1_ = getIValue("trans_dim1", match_vmap, vmap).value();
    if (!(trans_dim1_.isInt())) {
      return false;
    }
    auto flattern_shape_ = getIValue("flattern_shape", match_vmap, vmap).value();
    if (!(flattern_shape_.isInt())) {
      return false;
    }

    auto trans_dim0_val = trans_dim0_.toInt();
    auto trans_dim1_val = trans_dim1_.toInt();
    auto dim0_val = trans_dim0_val < trans_dim1_val ? trans_dim0_val : trans_dim1_val;
    auto dim1_val = trans_dim0_val > trans_dim1_val ? trans_dim0_val : trans_dim1_val;
    // If the tranpose if not for groups. ex. [n, c1, c2, h, w] => [n, c2, c1, h, w]
    if ((dim1_val - dim0_val) != 1) {
      return false;
    }

    auto input_val = input_.toTensor();
    auto view_shape_val = view_shape_.toIntVector();
    auto flattern_shape_val = flattern_shape_.toIntVector();
    // ex. [n, c, h, w] => [n, groups, c // groups, h, w]
    if ((input_val.ndimension() - view_shape_val.size()) != -1) {
      return false;
    }

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim0_val >= 0);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim1_val >= 0);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim0_val + 1 < input_val.ndimension());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim1_val + 1 < input_val.ndimension());
    if (view_shape_val[dim0_val] * view_shape_val[dim1_val] != input_val.size(dim0_val)) {
      return false;
    }

    if (flattern_shape_val.size() != input_val.ndimension()) {
      return false;
    }

    for (int i = 0; i < flattern_shape_val.size(); i++) {
      if (flattern_shape_val[i] != input_val.size(i)) {
        // [n, c, h, w] => view [n, groups, c // groups, h, w] => tranpose [n, c // groups, groups, h, w]
        // => view [n, -1, h, w]
        //    or
        //    view [n, c, h, w]
        if ((flattern_shape_val[i] != -1) || (i != dim0_val)) {
          return false;
        }
      }
    }

    return true;
  };

  IpexSubgraphRewriter rewriter_shuffle_2d;
  rewriter_shuffle_2d.RegisterRewritePattern(
    shuffle,
    shuffle_2d_fusion);
  rewriter_shuffle_2d.runOnGraph(graph);
}

void FuseAddLayerNorm(std::shared_ptr<Graph>& graph) {
  std::string aten_add_layernorm = R"(
      graph(%add_a, %add_b, %alpha, %shape:int[], %w, %b, %eps:float, %cudnn_enable:bool):
        %s = aten::add(%add_a, %add_b, %alpha)
        %r = aten::layer_norm(%s, %shape, %w, %b, %eps, %cudnn_enable)
        return (%r) )";
  std::string fused_add_layernorm = R"(
      graph(%add_a, %add_b, %alpha, %shape:int[], %w, %b, %eps:float, %cudnn_enable:bool):
        %r = ipex::add_layernorm(%add_a, %add_b, %alpha, %shape, %w, %b, %eps, %cudnn_enable)
        return (%r) )";
  SubgraphRewriter rewriter_aten;
  rewriter_aten.RegisterRewritePattern(aten_add_layernorm, fused_add_layernorm);
  rewriter_aten.runOnGraph(graph);
}

void FuseMHAScoreCalc(std::shared_ptr<Graph>& graph) {
  std::string div_matmul_add_softmax = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %_q = aten::div(%q, %dim_per_head)
        %qk = aten::matmul(%_q, %k)
        %_scores = aten::add(%qk, %relative_qk, %alpha)
        %scores = aten::softmax(%_scores, %softmax_dim, %dtype)
        return (%scores) )";

  std::string matmul_div_add_softmax = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %qk = aten::matmul(%q, %k)
        %_qk = aten::div(%qk, %dim_per_head)
        %_scores = aten::add(%_qk, %relative_qk, %alpha)
        %scores = aten::softmax(%_scores, %softmax_dim, %dtype)
        return (%scores) )";
  std::string div_matmul_add_softmax_fusion = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %scores = ipex::mha_scores_calc(%q, %k, %relative_qk, %alpha, %dim_per_head, %softmax_dim, %dtype)
        return (%scores) )";

  SubgraphRewriter mha_fusion;
  mha_fusion.RegisterRewritePattern(
      div_matmul_add_softmax, div_matmul_add_softmax_fusion);
  mha_fusion.RegisterRewritePattern(
      matmul_div_add_softmax, div_matmul_add_softmax_fusion);
  mha_fusion.runOnGraph(graph);
}

void replaceAtenMaxPool2dWithIpexMaxPool2d(std::shared_ptr<Graph>& graph) {
  std::string max_pool2d = R"(
      graph(%a, %kernel_size:int[], %stride:int[], %padding:int[], %dilation:int[], %ceil_mode:bool):
        %r = aten::max_pool2d(%a, %kernel_size, %stride, %padding, %dilation, %ceil_mode)
        return (%r) )";
  std::string ipex_max_pool2d = R"(
      graph(%a, %kernel_size:int[], %stride:int[], %padding:int[], %dilation:int[], %ceil_mode:bool):
        %r = ipex::max_pool2d(%a, %kernel_size, %stride, %padding, %dilation, %ceil_mode)
        return (%r) )";
  IpexSubgraphRewriter rewriter_max_pool2d;
  rewriter_max_pool2d.RegisterRewritePattern(max_pool2d, ipex_max_pool2d);
  rewriter_max_pool2d.runOnGraph(graph);
}

// replace aten::softmax to ipex::softmax during jit pass
// there is better performanc for ipex::softmax with oneDNN than aten::softmax
void replaceAtenSoftmaxWithIpexSoftmax(std::shared_ptr<Graph>& graph) {
  std::string aten_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = aten::softmax(%a, %dim, %half_to_float)
        return (%r) )";
  std::string ipex_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = ipex::softmax(%a, %dim, %half_to_float)
        return (%r) )";
  IpexSubgraphRewriter rewriter_aten;
  rewriter_aten.RegisterRewritePattern(aten_softmax, ipex_softmax);
  rewriter_aten.runOnGraph(graph);
}

void replaceEmbeddingBagWithQEmbeddingBag(std::shared_ptr<Graph> &graph) {
  std::string qembedingbag = R"(
     graph(%weight, %input, %offsets, %sparse, %include_last_offset, %o_scale, %o_zp, %o_dtype):
        %r = ipex::qembedding_bag(%weight, %input, %offsets, %sparse, %include_last_offset, %o_scale, %o_zp, %o_dtype)
        return (%r) )";

  std::string embeddingbag_with_quant_dequant = R"(
      graph(%qweight, %input, %offsets, %sparse, %include_last_offset,  %o_scale, %o_zp, %o_dtype):
        %dqw = aten::dequantize(%qweight)
        %r = torch_ipex::embedding_bag(%dqw, %input, %offsets, %sparse, %include_last_offset)
        %qout = aten::quantize_per_tensor(%r, %o_scale, %o_zp, %o_dtype)
        return (%qout) )";

  IpexSubgraphRewriter rewriter_qembeddingbag;
  rewriter_qembeddingbag.RegisterRewritePattern(embeddingbag_with_quant_dequant,
                                                qembedingbag);
  rewriter_qembeddingbag.runOnGraph(graph);
}

void replaceInteractionWithQInteraction(std::shared_ptr<Graph> &graph) {
  std::vector<std::string> patterns;
  std::vector<std::string> replacements;
  std::string graph_common_head = R"(graph()";
  std::string graph_common_tail = R"(, %o_scale, %o_zp, %o_dtype):
  )";
  std::string list_construct_common_head =
      R"(%input : Tensor[] = prim::ListConstruct()";
  std::string list_construct_common_tail = R"() )";
  std::string replacement_common_tail =
      R"(%out =  ipex::qinteraction(%input, %o_scale, %o_zp, %o_dtype) return (%out) )";
  std::string pattern_common_tail =
      R"(%out = torch_ipex::interaction_forward(%input)  %qout = aten::quantize_per_tensor(%out, %o_scale, %o_zp, %o_dtype) return (%qout) )";

  for (auto *n : graph->block()->nodes()) {
    if (n->kind() ==
        Symbol::fromQualString("torch_ipex::interaction_forward")) {
      size_t id = 0;
      auto ListConstructNode = n->input(0)->node();

      bool is_quantized =
          std::any_of(ListConstructNode->inputs().begin(),
                      ListConstructNode->inputs().end(), [](auto &v) {
                        return v->node()->kind() == Symbol::aten("dequantize");
                      });

      if (!is_quantized)
        return;

      std::string pattern = R"()";
      std::string replacement = R"()";
      std::string dequantizes = R"()";
      std::vector<std::string> qinputs;
      std::vector<std::string> dqinputs;
      for (auto input : ListConstructNode->inputs()) {
        if (input->node()->kind() == Symbol::aten("dequantize")) {
          qinputs.push_back("%q" + std::to_string(id));
          dqinputs.push_back("%dq" + std::to_string(id));
          std::string dequantize = "%dq" + std::to_string(id) +
                                   " : Tensor = aten::dequantize(" + "%q" +
                                   std::to_string(id) + ")";
          dequantizes.append(dequantize);
          ++id;
        }
      }

      std::string header =
          graph_common_head + c10::Join(", ", qinputs) + graph_common_tail;
      pattern += header;
      pattern += dequantizes;
      pattern += list_construct_common_head + c10::Join(", ", dqinputs) +
                 list_construct_common_tail;
      pattern += pattern_common_tail;
      patterns.push_back(pattern);

      replacement = header;
      replacement += list_construct_common_head + c10::Join(", ", qinputs) +
                     list_construct_common_tail;
      replacement += replacement_common_tail;
      replacements.push_back(replacement);
    }
  }

  IpexSubgraphRewriter rewriter;
  for (size_t i = 0; i < patterns.size(); i++) {
    rewriter.RegisterRewritePattern(patterns[i], replacements[i]);
    rewriter.runOnGraph(graph);
  }
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch

