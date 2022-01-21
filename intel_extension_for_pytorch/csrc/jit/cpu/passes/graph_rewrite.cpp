
#include "graph_rewrite.h"
#include <torch/csrc/jit/passes/remove_mutation.h>
#include "utils.h"

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

  auto filter_shuffle_2d_fusion =
      [](const Match& match,
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
        auto flattern_shape_ =
            getIValue("flattern_shape", match_vmap, vmap).value();
        if (!(flattern_shape_.isInt())) {
          return false;
        }

        auto trans_dim0_val = trans_dim0_.toInt();
        auto trans_dim1_val = trans_dim1_.toInt();
        auto dim0_val =
            trans_dim0_val < trans_dim1_val ? trans_dim0_val : trans_dim1_val;
        auto dim1_val =
            trans_dim0_val > trans_dim1_val ? trans_dim0_val : trans_dim1_val;
        // If the tranpose if not for groups. ex. [n, c1, c2, h, w] => [n, c2,
        // c1, h, w]
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
        if (view_shape_val[dim0_val] * view_shape_val[dim1_val] !=
            input_val.size(dim0_val)) {
          return false;
        }

        if (flattern_shape_val.size() != input_val.ndimension()) {
          return false;
        }

        for (int i = 0; i < flattern_shape_val.size(); i++) {
          if (flattern_shape_val[i] != input_val.size(i)) {
            // [n, c, h, w] => view [n, groups, c // groups, h, w] => tranpose
            // [n, c // groups, groups, h, w]
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

  SubgraphRewriter rewriter_shuffle_2d;
  rewriter_shuffle_2d.RegisterRewritePattern(shuffle, shuffle_2d_fusion);
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

// MHA fusion covers aten::softmax, ipex::softmax and ipex::softmax_:
// (1) MHA obviously shows better performance than aten div/matmul/add/softmax.
// (2) MHA also shows better performance than aten add + matmul_div fusion
//     + ipex::softmax/softmax_.
// (3) Current ipex::softmax/softmax_ is from the replacement of aten::softmax,
//     it is safe to make MHA cover ipex::softmax/softmax_.
void FuseMHAScoreCalc(std::shared_ptr<Graph>& graph) {
  std::string div_matmul_add_aten_softmax = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %_q = aten::div(%q, %dim_per_head)
        %qk = aten::matmul(%_q, %k)
        %_scores = aten::add(%qk, %relative_qk, %alpha)
        %scores = aten::softmax(%_scores, %softmax_dim, %dtype)
        return (%scores) )";

  std::string div_matmul_add_ipex_softmax = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %_q = aten::div(%q, %dim_per_head)
        %qk = aten::matmul(%_q, %k)
        %_scores = aten::add(%qk, %relative_qk, %alpha)
        %scores = ipex::softmax(%_scores, %softmax_dim, %dtype)
        return (%scores) )";

  std::string div_matmul_add_ipex_softmax_ = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %_q = aten::div(%q, %dim_per_head)
        %qk = aten::matmul(%_q, %k)
        %_scores = aten::add(%qk, %relative_qk, %alpha)
        %scores = ipex::softmax_(%_scores, %softmax_dim, %dtype)
        return (%scores) )";

  std::string matmul_div_add_aten_softmax = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %qk = aten::matmul(%q, %k)
        %_qk = aten::div(%qk, %dim_per_head)
        %_scores = aten::add(%_qk, %relative_qk, %alpha)
        %scores = aten::softmax(%_scores, %softmax_dim, %dtype)
        return (%scores) )";

  std::string matmul_div_add_ipex_softmax = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %qk = aten::matmul(%q, %k)
        %_qk = aten::div(%qk, %dim_per_head)
        %_scores = aten::add(%_qk, %relative_qk, %alpha)
        %scores = ipex::softmax(%_scores, %softmax_dim, %dtype)
        return (%scores) )";

  std::string matmul_div_add_ipex_softmax_ = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %qk = aten::matmul(%q, %k)
        %_qk = aten::div(%qk, %dim_per_head)
        %_scores = aten::add(%_qk, %relative_qk, %alpha)
        %scores = ipex::softmax_(%_scores, %softmax_dim, %dtype)
        return (%scores) )";

  std::string div_matmul_add_softmax_fusion = R"(
      graph(%q:Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %scores = ipex::mha_scores_calc(%q, %k, %relative_qk, %alpha, %dim_per_head, %softmax_dim, %dtype)
        return (%scores) )";

  SubgraphRewriter mha_fusion;
  mha_fusion.RegisterRewritePattern(
      div_matmul_add_aten_softmax, div_matmul_add_softmax_fusion);
  mha_fusion.RegisterRewritePattern(
      div_matmul_add_ipex_softmax, div_matmul_add_softmax_fusion);
  mha_fusion.RegisterRewritePattern(
      div_matmul_add_ipex_softmax_, div_matmul_add_softmax_fusion);
  mha_fusion.RegisterRewritePattern(
      matmul_div_add_aten_softmax, div_matmul_add_softmax_fusion);
  mha_fusion.RegisterRewritePattern(
      matmul_div_add_ipex_softmax, div_matmul_add_softmax_fusion);
  mha_fusion.RegisterRewritePattern(
      matmul_div_add_ipex_softmax_, div_matmul_add_softmax_fusion);
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
  SubgraphRewriter rewriter_max_pool2d;
  rewriter_max_pool2d.RegisterRewritePattern(max_pool2d, ipex_max_pool2d);
  rewriter_max_pool2d.runOnGraph(graph);
}

// for contiguous input:
// replace aten::softmax to ipex::softmax/ipex::softmax_ during jit pass
// there is better performance for ipex::softmax/ipex::softmax_ with oneDNN than
// aten::softmax
// for non-contiguous input:
// (1) oneDNN will use ref path which is not optimized as expected
// (2) if do contiguous copy then go into oneDNN optimized path, the
// copy overhead is unneglectable
// (3) so here will not replace aten::softmax to avoid unexpected regression
void replaceAtenSoftmaxWithIpexSoftmax(std::shared_ptr<Graph>& graph) {
  std::string aten_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = aten::softmax(%a, %dim, %half_to_float)
        return (%r) )";
  std::string ipex_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = ipex::softmax(%a, %dim, %half_to_float)
        return (%r) )";
  std::string ipex_softmax_ = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = ipex::softmax_(%a, %dim, %half_to_float)
        return (%r) )";

  // Filter the unsupported case for inplace softmax
  auto filter_inplace =
      [graph](
          const Match& match,
          const std::unordered_map<std::string, Value*>& vmap) {
        Node* node = match.anchor;
        std::unique_ptr<AliasDb> aliasDb_ = std::make_unique<AliasDb>(graph);

        // check if the input is contiguous, and skip if it is not
        auto input_value = node->input(0)->type()->cast<TensorType>();
        auto input_value_contiguous = input_value->contiguous();
        bool is_contiguous =
            input_value_contiguous->strides() == input_value->strides();
        if (!is_contiguous) {
          return false;
        }

        // Skip if input has more than one use
        if (node->input(0)->uses().size() > 1) {
          return false;
        }
        // Skip if input's def node has side effect or input has alias
        if (MutationRemover::hasSideEffectOrAlias(
                node->inputs().at(0), aliasDb_.get())) {
          return false;
        }
        return true;
      };

  auto filter_outplace =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        Node* node = match.anchor;
        // check if the input is contiguous, and skip if it is not
        auto input_value = node->input(0)->type()->cast<TensorType>();
        auto input_value_contiguous = input_value->contiguous();
        bool is_contiguous =
            input_value_contiguous->strides() == input_value->strides();
        if (!is_contiguous) {
          return false;
        }
        return true;
      };

  // try to replace inplace softmax first
  SubgraphRewriter rewriter_aten_inplace;
  rewriter_aten_inplace.RegisterRewritePattern(aten_softmax, ipex_softmax_);
  rewriter_aten_inplace.runOnGraph(graph, filter_inplace);
  // if any miss, then try to replace outplace softmax
  SubgraphRewriter rewriter_aten;
  rewriter_aten.RegisterRewritePattern(aten_softmax, ipex_softmax);
  rewriter_aten.runOnGraph(graph, filter_outplace);
}

void replaceAtenBatchNormWithIpexBatchNorm(std::shared_ptr<Graph>& graph) {
  std::string batch_norm = R"(
      graph(%a, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled):
        %r = aten::batch_norm(%a, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled)
        return (%r) )";
  std::string ipex_batch_norm = R"(
      graph(%a, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled):
        %r = ipex::batch_norm(%a, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled)
        return (%r) )";

  SubgraphRewriter rewriter_batch_norm;
  rewriter_batch_norm.RegisterRewritePattern(batch_norm, ipex_batch_norm);
  rewriter_batch_norm.runOnGraph(graph);
}

void replaceEmbeddingBagWithQEmbeddingBag(std::shared_ptr<Graph>& graph) {
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

  SubgraphRewriter rewriter_qembeddingbag;
  rewriter_qembeddingbag.RegisterRewritePattern(
      embeddingbag_with_quant_dequant, qembedingbag);
  rewriter_qembeddingbag.runOnGraph(graph);
}

void replaceInteractionWithQInteraction(std::shared_ptr<Graph>& graph) {
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

  for (auto* n : graph->block()->nodes()) {
    if (n->kind() ==
        Symbol::fromQualString("torch_ipex::interaction_forward")) {
      size_t id = 0;
      auto ListConstructNode = n->input(0)->node();

      bool is_quantized = std::any_of(
          ListConstructNode->inputs().begin(),
          ListConstructNode->inputs().end(),
          [](auto& v) {
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
              " : Tensor = aten::dequantize(" + "%q" + std::to_string(id) + ")";
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

  SubgraphRewriter rewriter;
  for (size_t i = 0; i < patterns.size(); i++) {
    rewriter.RegisterRewritePattern(patterns[i], replacements[i]);
    rewriter.runOnGraph(graph);
  }
}

void FuseConcatBnRelu(std::shared_ptr<Graph>& graph) {
  std::string aten_concat_bn_relu = R"(
      graph(%input : Tensor[], %dim:int, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled):
        %a = aten::cat(%input, %dim)
        %b = aten::batch_norm(%a, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled)
        %c = aten::relu(%b)
        return (%c) )";
  std::string fused_concat_bn_relu = R"(
      graph(%input : Tensor[], %dim:int, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled):
        %alpha: int = prim::Constant[value=1]()
        %u1 = aten::add(%running_var, %eps, %alpha)
        %u2 = aten::sqrt(%u1)
        %u3 = aten::div(%running_mean, %u2)
        %u4 = aten::mul(%weight, %u3)
        %beta = aten::sub(%bias, %u4, %alpha)
        %b = ipex::concat_bn_relu(%input, %beta, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled, %dim)
        return (%b) )";

  auto fusion_filter = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    Node* node = match.anchor;
    const auto& match_vmap = match.values_map;
    // Check if the Concat Dimension is the channel
    auto dim_ = getIValue("dim", match_vmap, vmap).value();
    if (!(dim_.isInt())) {
      return false;
    }
    auto dim = dim_.toInt();
    if (dim != 1) {
      return false;
    }
    // Find the Concat node
    auto n = node->input(0)->node()->input(0)->node();
    TORCH_CHECK(n->kind() == aten::cat);

    auto listConstruct = n->input(0)->node();
    int64_t input_len = 0;
    for (auto p : listConstruct->inputs()) {
      input_len++;
    }
    auto tensor1 = listConstruct->input(0)->type()->cast<TensorType>();
    auto check_type_channelsize = [](std::shared_ptr<c10::TensorType> tensor) {
      return (
          tensor->scalarType().value() == at::kFloat &&
          tensor->sizes()[1].value() % 16 == 0 && is_channelslast(tensor));
    };
    // Check if the dimension of the first tensor is either 4 or 5.
    // Check if the data type, the size of Channels, and the memory format are
    // float, mutiples of 16, and ChannelsLast(3d), respectively.
    if (!(tensor1->dim().value() == 4 || tensor1->dim().value() == 5) ||
        !check_type_channelsize(tensor1)) {
      return false;
    }
    // Check the rest tensors
    for (int64_t i = 1; i < input_len; ++i) {
      auto tensori = listConstruct->input(i)->type()->cast<TensorType>();
      // Check dimension, data type, channel size and memory format
      if (!(tensor1->dim().value() == tensori->dim().value()) ||
          !check_type_channelsize(tensori)) {
        return false;
      }
      // The channel sizes can be different, and check the other dim sizes.
      for (int64_t j = 0; j < tensori->dim().value(); ++j) {
        if (j != 1 &&
            tensor1->sizes()[j].value() != tensori->sizes()[j].value()) {
          return false;
        }
      }
    }
    return true;
  };

  SubgraphRewriter rewriter_concatbnrelu;
  rewriter_concatbnrelu.RegisterRewritePattern(
      aten_concat_bn_relu, fused_concat_bn_relu);
  rewriter_concatbnrelu.runOnGraph(graph, fusion_filter);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch