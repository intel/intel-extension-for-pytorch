#include "graph_rewrite.h"
#include "graph_rewrite_helper.h"
#include "utils.h"

#include <ATen/code_template.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include "utils/onednn_utils.h"

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

using namespace torch::jit;

// FuseShuffle is matching the channelshuffle pattern, where:
// (1) the first view is [n, c, h, w] => [n, groups, c // groups, h, w]
// (2) the tranpose is for groups => [n, c // groups, grpups, h, w]
// (3) the output view shape should be the same as the input tensor shape
void FuseShuffle(std::shared_ptr<Graph>& graph) {
  // below is channelshuffle for staic view shape pattern
  std::string channelshuffle_with_static_shape = R"(
      graph(%input, %view_shape:int[], %trans_dim0:int, %trans_dim1:int, %mem_format:int, %flattern_shape:int[]):
        %r1 = aten::view(%input, %view_shape)
        %r2 = aten::transpose(%r1, %trans_dim0, %trans_dim1)
        %r3 = aten::contiguous(%r2, %mem_format)
        %r4 = aten::view(%r3, %flattern_shape)
        return (%r4) )";

  std::string shuffle_2d_fusion_with_static_shape = R"(
      graph(%input, %view_shape:int[], %trans_dim0:int, %trans_dim1:int, %mem_format:int, %flattern_shape:int[]):
        %r = ipex::shuffle_2d(%input, %view_shape, %trans_dim0, %trans_dim1)
        return (%r) )";

  // below is channelshuffle for dynamic view shape pattern
  std::string dynamic_shape_input = R"(
      graph(%input, %idx_0:int, %idx_1:int, %idx_2:int, %idx_3:int, %div_g, %g:int, %flattern_c):
        %n_ = aten::size(%input, %idx_0)
        %c_ = aten::size(%input, %idx_1)
        %tensor_c_ = prim::NumToTensor(%c_)
        %h_ = aten::size(%input, %idx_2)
        %w_ = aten::size(%input, %idx_3)
        %c_div_g_ = aten::floor_divide(%tensor_c_, %div_g)
        %int_c_div_g_ = aten::Int(%c_div_g_)
        %view_shape:int[] = prim::ListConstruct(%n_, %g, %int_c_div_g_, %h_, %w_) )";

  std::string channelshuffle_for_dynamic_shape = R"(
        %r1 = aten::view(%input, %view_shape)
        %r2 = aten::transpose(%r1, %idx_1, %idx_2)
        %r3 = aten::contiguous(%r2, %idx_0)
        %flattern_shape:int[] = prim::ListConstruct(%n_, %flattern_c, %h_, %w_)
        %r4 = aten::view(%r3, %flattern_shape)
        return (%r4) )";

  std::string shuffle_2d_fusion_for_dynamic_shape = R"(
        %r = ipex::shuffle_2d(%input, %view_shape, %idx_1, %idx_2)
        return (%r) )";

  std::string channelshuffle_with_dynamic_shape =
      dynamic_shape_input + channelshuffle_for_dynamic_shape;
  std::string shuffle_2d_fusion_with_dynamic_shape =
      dynamic_shape_input + shuffle_2d_fusion_for_dynamic_shape;

  auto filter_shuffle_2d_static_fusion =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        const auto& match_vmap = match.values_map;

        // current node is the second "view" node
        Node* node = match.anchor;
        // get the first "view" node
        auto first_view_node =
            node->input(0)->node()->input(0)->node()->input(0)->node();
        // get the input tensor from the first "view" node
        auto input_ = first_view_node->input(0);
        auto inputType = input_->type()->cast<TensorType>();
        auto inputTensor = *inputType;
        // if the input tensor does not have dim info
        if (!inputType->dim().has_value()) {
          return false;
        }
        // get the view shape
        auto view_shape_ = first_view_node->input(1);
        // get the flattern shape
        auto flattern_shape_ = node->input(1);
        // get the transpose node
        auto trans_node = node->input(0)->node()->input(0)->node();

        auto trans_dim0_ = trans_node->input(1);
        auto trans_dim1_ = trans_node->input(2);
        // if the transpose dim has not set
        if (!toIValue(trans_dim0_).has_value() ||
            !toIValue(trans_dim1_).has_value()) {
          return false;
        }
        auto trans_dim0_val = toIValue(trans_dim0_).value().toInt();
        auto trans_dim1_val = toIValue(trans_dim1_).value().toInt();
        auto dim0_val =
            trans_dim0_val < trans_dim1_val ? trans_dim0_val : trans_dim1_val;
        auto dim1_val =
            trans_dim0_val > trans_dim1_val ? trans_dim0_val : trans_dim1_val;
        // If the tranpose if not for groups. ex. [n, c1, c2, h, w] => [n, c2,
        // c1, h, w]
        if ((dim1_val - dim0_val) != 1) {
          return false;
        }

        // if the view shape or flattern shape is not set
        if (!toIValue(view_shape_).has_value() ||
            !toIValue(flattern_shape_).has_value()) {
          return false;
        }

        auto view_shape_list = toIValue(view_shape_).value().toIntVector();
        auto flattern_shape_list =
            toIValue(flattern_shape_).value().toIntVector();

        // ex. [n, c, h, w] => [n, groups, c // groups, h, w]
        if ((inputType->dim().value() - view_shape_list.size()) != -1) {
          return false;
        }

        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim0_val >= 0);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim1_val >= 0);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            dim0_val + 1 < inputType->dim().value());
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            dim1_val + 1 < inputType->dim().value());

        auto view_shape_dim0_val = view_shape_list[dim0_val];
        auto view_shape_dim1_val = view_shape_list[dim1_val];
        // c => groups, c // groups
        if (view_shape_dim0_val * view_shape_dim1_val !=
            inputTensor.sizes()[dim0_val].value()) {
          return false;
        }
        // output view shape should be the same as the input
        if (flattern_shape_list.size() != inputType->dim().value()) {
          return false;
        }

        for (int i = 0; i < flattern_shape_list.size(); i++) {
          if (flattern_shape_list[i] != inputTensor.sizes()[i].value()) {
            // [n, c, h, w] => view [n, groups, c // groups, h, w] => tranpose
            // [n, c // groups, groups, h, w]
            // => view [n, -1, h, w]
            //    or
            //    view [n, c, h, w]
            if ((flattern_shape_list[i] != -1) || (i != dim0_val)) {
              return false;
            }
          }
        }
        return true;
      };

  auto filter_shuffle_2d_dynamic_fusion =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        const auto& match_vmap = match.values_map;

        auto n_idx = torch_ipex::jit::graph_rewrite_helper::getIValue(
            "idx_0", match_vmap, vmap);
        auto c_idx = torch_ipex::jit::graph_rewrite_helper::getIValue(
            "idx_1", match_vmap, vmap);
        auto h_idx = torch_ipex::jit::graph_rewrite_helper::getIValue(
            "idx_2", match_vmap, vmap);
        auto w_idx = torch_ipex::jit::graph_rewrite_helper::getIValue(
            "idx_3", match_vmap, vmap);
        if (!n_idx.has_value() || !c_idx.has_value() || !h_idx.has_value() ||
            !w_idx.has_value()) {
          return false;
        }

        auto n_idx_ = n_idx.value().toInt();
        auto c_idx_ = c_idx.value().toInt();
        auto h_idx_ = h_idx.value().toInt();
        auto w_idx_ = w_idx.value().toInt();

        if ((n_idx_ != 0) || (c_idx_ != 1) || (h_idx_ != 2) || (w_idx_ != 3)) {
          return false;
        }

        return true;
      };

  SubgraphRewriter rewriter_shuffle_2d_dynamic;
  rewriter_shuffle_2d_dynamic.RegisterRewritePattern(
      channelshuffle_with_dynamic_shape, shuffle_2d_fusion_with_dynamic_shape);
  rewriter_shuffle_2d_dynamic.runOnGraph(
      graph, filter_shuffle_2d_dynamic_fusion);
  SubgraphRewriter rewriter_shuffle_2d_static;
  rewriter_shuffle_2d_static.RegisterRewritePattern(
      channelshuffle_with_static_shape, shuffle_2d_fusion_with_static_shape);
  rewriter_shuffle_2d_static.runOnGraph(graph, filter_shuffle_2d_static_fusion);

} // namespace graph_rewrite

void FuseRMSNorm(std::shared_ptr<Graph>& graph) {
  std::string aten_RMSNorm = R"(
      graph(%hidden_states, %weight, %exponent:int, %dim:int[], %keepdim:bool, %dtype:NoneType, %eps:float, %alpha:int):
        %s = aten::pow(%hidden_states, %exponent)
        %v = aten::mean(%s, %dim, %keepdim, %dtype)
        %m = aten::add(%v, %eps, %alpha)
        %n = aten::rsqrt(%m)
        %l = aten::mul(%hidden_states, %n)
        %r = aten::mul(%weight, %l)
        return (%r) )";
  std::string fused_RMSNorm = R"(
      graph(%hidden_states, %weight, %exponent:int, %dim:int[], %keepdim:bool, %dtype:NoneType, %eps:float, %alpha:int):
        %r = ipex::RMSNorm(%hidden_states, %weight, %eps)
        return (%r) )";
  SubgraphRewriter rewriter_aten;
  rewriter_aten.RegisterRewritePattern(aten_RMSNorm, fused_RMSNorm);
  rewriter_aten.runOnGraph(graph);
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

void FuseMatmulDivOrMul(std::shared_ptr<Graph>& graph) {
  const std::string div_str = R"(div)";
  const std::string div_inplace_str = R"(div_)";
  std::vector<std::string> div_ops = {div_str, div_inplace_str};
  const std::string mul_str = R"(mul)";
  const std::string mul_inplace_str = R"(mul_)";
  std::vector<std::string> mul_ops = {mul_str, mul_inplace_str};

  auto aten_div_pattern = at::jit::CodeTemplate(R"(
      graph(%x, %y, %z):
        %mm_res = aten::matmul(%x, %y)
        %div_res = aten::${div_op}(%mm_res, %z)
        return (%div_res) )");

  auto aten_div_pattern_with_out = at::jit::CodeTemplate(R"(
      graph(%x, %y, %z, %out):
        %mm_res = aten::matmul(%x, %y, %out)
        %div_res = aten::${div_op}(%mm_res, %z)
        return (%div_res) )");

  auto aten_mul_pattern = at::jit::CodeTemplate(R"(
      graph(%x, %y, %z):
        %mm_res = aten::matmul(%x, %y)
        %mul_res = aten::${mul_op}(%mm_res, %z)
        return (%mul_res) )");

  auto aten_mul_pattern_with_out = at::jit::CodeTemplate(R"(
      graph(%x, %y, %z, %out):
        %mm_res = aten::matmul(%x, %y, %out)
        %mul_res = aten::${mul_op}(%mm_res, %z)
        return (%mul_res) )");

  std::string fused_matmul_div = R"(
      graph(%x, %y, %z):
        %r = ipex::matmul_div(%x, %y, %z)
        return (%r) )";
  std::string fused_matmul_div_with_out = R"(
      graph(%x, %y, %z, %out):
        %r = ipex::matmul_div(%x, %y, %out, %z)
        return (%r) )";
  std::string fused_matmul_mul = R"(
      graph(%x, %y, %z):
        %r = ipex::matmul_mul(%x, %y, %z)
        return (%r) )";
  std::string fused_matmul_mul_with_out = R"(
      graph(%x, %y, %z, %out):
        %r = ipex::matmul_mul(%x, %y, %out, %z)
        return (%r) )";
  auto filter_scalar = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    Node* node = match.anchor;
    auto target_value = node->input(1);
    return utils::is_scalar(target_value);
  };

  for (auto const& it : div_ops) {
    at::jit::TemplateEnv env;
    env.s("div_op", it);

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        aten_div_pattern.format(env), fused_matmul_div);
    rewriter.RegisterRewritePattern(
        aten_div_pattern_with_out.format(env), fused_matmul_div_with_out);
    rewriter.runOnGraph(graph);
  }
  for (auto const& it : mul_ops) {
    at::jit::TemplateEnv env;
    env.s("mul_op", it);

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        aten_mul_pattern.format(env), fused_matmul_mul);
    rewriter.RegisterRewritePattern(
        aten_mul_pattern_with_out.format(env), fused_matmul_mul_with_out);
    rewriter.runOnGraph(graph, filter_scalar);
  }
}

void PostScalarDivOrMul(std::shared_ptr<Graph>& graph) {
  std::string div_matmul = R"(
      graph(%q: Tensor, %k: Tensor, %dim_per_head:float): 
        %_q = aten::div(%q, %dim_per_head)
        %r = aten::matmul(%_q, %k) 
        return (%r) )";

  std::string mul_matmul = R"(
      graph(%q: Tensor, %k: Tensor, %scale:float): 
        %_q = aten::mul(%q, %scale)
        %r = aten::matmul(%_q, %k) 
        return (%r) )";

  std::string matmul_div = R"(
      graph(%q: Tensor, %k: Tensor, %dim_per_head:float): 
        %qk = aten::matmul(%q, %k) 
        %r = aten::div(%qk, %dim_per_head)
        return (%r) )";

  std::string matmul_mul = R"(
      graph(%q: Tensor, %k: Tensor, %scale:float): 
        %qk = aten::matmul(%q, %k) 
        %r = aten::mul(%qk, %scale)
        return (%r) )";
  auto filter_scalar = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    Node* node = match.anchor;
    auto target_value = node->input(0)->node()->input(1);
    return utils::is_scalar(target_value);
  };
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(div_matmul, matmul_div);
  rewriter.RegisterRewritePattern(mul_matmul, matmul_mul);
  rewriter.runOnGraph(graph, filter_scalar);
}

// MHA fusion covers aten::softmax, ipex::softmax and ipex::softmax_:
// (1) MHA obviously shows better performance than aten div/matmul/add/softmax.
// (2) MHA also shows better performance than aten add + matmul_div fusion
//     + ipex::softmax/softmax_.
// (3) Current ipex::softmax/softmax_ is from the replacement of aten::softmax,
//     it is safe to make MHA cover ipex::softmax/softmax_.
// Note that we have a prepass (PostScalarDivOrMul) to move scalar div/mul after
// matmul so that we can match the uniform matmul+div/mul pattern
void FuseMHAScoreCalc(std::shared_ptr<Graph>& graph) {
  // below are basic patterns string for MHA matching
  std::string args_matmul_div_add_softmax = R"(
      graph(%q: Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype): )";

  std::string args_matmul_mul_add_softmax = R"(
      graph(%q: Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %scale:float, %softmax_dim:int, %dtype): )";

  std::string args_matmul_mul_maskedfill_softmax = R"(
      graph(%q: Tensor, %k: Tensor, %_mask_qk_shape: Tensor, %dim_per_head:int, %fill:float, %softmax_dim:int, %dtype): )";

  std::string args_expand_maskedfill_softmax = R"(
      graph(%qk: Tensor, %mask_qk: Tensor, %mask_qk_reshp: int[], %fill:float, %softmax_dim:int, %dtype): )";

  std::string args_matmul_div_expand_maskedfill_softmax = R"(
      graph(%q: Tensor, %k: Tensor, %mask_qk: Tensor, %mask_qk_reshp: int[], %fill:float, %dim_per_head:float, %softmax_dim:int, %dtype): )";

  std::string matmul_mul_add = R"(
        %_qk = aten::matmul(%q, %k)
        %qk = aten::mul(%_qk, %scale)
        %_scores = aten::add(%qk, %relative_qk, %alpha) )";

  std::string matmul_div_add = R"(
        %qk = aten::matmul(%q, %k)
        %_qk = aten::div(%qk, %dim_per_head)
        %_scores = aten::add(%_qk, %relative_qk, %alpha) )";

  std::string matmul_mul = R"(
        %_qk = aten::matmul(%q, %k)
        %qk = aten::mul(%_qk, %dim_per_head) )";

  std::string matmul_div = R"(
        %_qk = aten::matmul(%q, %k)
        %qk = aten::div(%_qk, %dim_per_head) )";

  std::string expand = R"(
        %_mask_qk_view = aten::view(%mask_qk, %mask_qk_reshp)
        %_mask_qk_shape = aten::expand_as(%_mask_qk_view, %qk)  )";

  std::string aten_masked_fill = R"(
        %_scores = aten::masked_fill(%qk, %_mask_qk_shape, %fill) )";

  std::string aten_masked_fill_ = R"(
        %_scores = aten::masked_fill_(%qk, %_mask_qk_shape, %fill) )";

  std::string aten_softmax = R"(
        %scores = aten::softmax(%_scores, %softmax_dim, %dtype) )";

  std::string set_return = R"(
        return (%scores) )";

  // below are MHA score combinations from Bert Model (matmul+div+add+softmax)
  std::string matmul_div_add_softmax =
      args_matmul_div_add_softmax + matmul_div_add + aten_softmax + set_return;
  std::string matmul_mul_add_softmax =
      args_matmul_mul_add_softmax + matmul_mul_add + aten_softmax + set_return;

  // below are MHA score combinations from DistilBert Model
  // (matmul+div+expand+masked_fill+softmax)
  // Note: here and below we match both inplace masked_fill_ and outplace
  // masked_fill, since there could be cases (e.g., from DistilBert model) that
  // the qk scores to be maked_fill have multi-use, like using scores tensor to
  // generate the mask. Then it makes masked_fill_ pattern necessary since we
  // are unable to convert it into the outplace one at first for a uniform
  // pattern matching.
  std::string matmul_div_maskedfill__softmax =
      args_matmul_div_expand_maskedfill_softmax + matmul_div + expand +
      aten_masked_fill_ + aten_softmax + set_return;
  std::string matmul_div_maskedfill_softmax =
      args_matmul_div_expand_maskedfill_softmax + matmul_div + expand +
      aten_masked_fill + aten_softmax + set_return;

  // below are parts of MHA score combinations from DistilBert Model (for int8
  // path) (expand+masked_fill+softmax)
  std::string expand_maskedfill__softmax = args_expand_maskedfill_softmax +
      expand + aten_masked_fill_ + aten_softmax + set_return;
  std::string expand_maskedfill_softmax = args_expand_maskedfill_softmax +
      expand + aten_masked_fill + aten_softmax + set_return;

  // below are MHA score combinations from ats_vit Model
  // (matmul+mul+masked_fill+softmax)
  std::string matmul_mul_maskedfill__softmax =
      args_matmul_mul_maskedfill_softmax + matmul_mul + aten_masked_fill_ +
      aten_softmax + set_return;
  std::string matmul_mul_maskedfill_softmax =
      args_matmul_mul_maskedfill_softmax + matmul_mul + aten_masked_fill +
      aten_softmax + set_return;

  // below are fusion patterns
  std::string matmul_div_add_softmax_fusion = R"(
      graph(%q: Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %dim_per_head:int, %softmax_dim:int, %dtype):
        %scores = ipex::mha_scores_calc(%q, %k, %relative_qk, %alpha, %dim_per_head, %softmax_dim, %dtype)
        return (%scores) )";

  std::string matmul_mul_add_softmax_fusion = R"(
      graph(%q: Tensor, %k: Tensor, %relative_qk: Tensor, %alpha:int, %scale:float, %softmax_dim:int, %dtype):
        %scores = ipex::mha_scores_calc_v2(%q, %k, %relative_qk, %alpha, %scale, %softmax_dim, %dtype)
        return (%scores) )";

  std::string matmul_div_maskedfill_softmax_fusion = R"(

      graph(%q: Tensor, %k: Tensor, %mask_qk: Tensor, %mask_qk_reshp: int[], %fill:float, %dim_per_head:float, %softmax_dim:int, %dtype):
        %scores = ipex::distil_mha_scores_calc(%q, %k, %mask_qk, %mask_qk_reshp, %fill, %dim_per_head)
        return (%scores) )";

  std::string expand_maskedfill_softmax_fusion = R"(
      graph(%qk: Tensor, %mask_qk: Tensor, %mask_qk_reshp: int[], %fill:float,  %softmax_dim:int, %dtype):
        %scores = ipex::maskedfill_softmax(%qk, %mask_qk, %mask_qk_reshp, %fill)
        return (%scores) )";

  std::string matmul_mul_maskedfill_softmax_fusion = R"(
      graph(%q: Tensor, %k: Tensor, %_mask_qk_shape: Tensor, %dim_per_head:int, %fill:float, %softmax_dim:int, %dtype):
        %scores = ipex::vit_mha_scores_calc(%q, %k, %_mask_qk_shape, %fill, %dim_per_head)
        return (%scores) )";

  auto filter_common = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;

    // Only support last dimension for softmax
    auto dim_ = torch_ipex::jit::graph_rewrite_helper::getIValue(
        "softmax_dim", match_vmap, vmap);
    if (!dim_.has_value() || !(dim_.value().isInt())) {
      return false;
    }
    auto dim = dim_.value().toInt();
    if (dim != -1) {
      return false;
    }
    Node* node = match.anchor;
    // Find the masked_fill node to get the qk value
    auto qk_node = node->input(0)->node();
    TORCH_CHECK(
        qk_node->kind() == aten::masked_fill ||
        qk_node->kind() == aten::masked_fill_);
    // // Only support contiguous tensor for qk
    auto qk_value = qk_node->input(0)->type()->cast<TensorType>();
    if (!utils::is_contiguous(qk_value)) {
      return false;
    }

    // Only support qk.dim >=2D
    if (!qk_value->dim().has_value() || qk_value->dim().value() < 2) {
      return false;
    }

    // Checking the dtype as None
    auto dtype_value = torch_ipex::jit::graph_rewrite_helper::getIValue(
        "dtype", match_vmap, vmap);

    if (!dtype_value.has_value() || !dtype_value.value().isNone()) {
      return false;
    }

    return true;
  };

  auto filter_distil_mha =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        Node* node = match.anchor;
        // Find the masked_fill node
        auto qk_node = node->input(0)->node();
        TORCH_CHECK(
            qk_node->kind() == aten::masked_fill ||
            qk_node->kind() == aten::masked_fill_);

        // Check whether the fill value of masked_fill is constant
        // This constant fill value could be either 0-dim tensor or just a
        // scalar
        auto fill_value_node = qk_node->input(2)->node();
        if (fill_value_node->kind() != prim::Constant ||
            !utils::is_scalar(qk_node->input(2))) {
          return false;
        }

        // Find the view node to get the mask value
        auto mask_node =
            node->input(0)->node()->input(1)->node()->input(0)->node();
        TORCH_CHECK(mask_node->kind() == aten::view);
        auto mask_value = mask_node->input(0)->type()->cast<TensorType>();
        // Only support contiguous tensor for mask
        if (!utils::is_contiguous(mask_value)) {
          return false;
        }

        // Only support when expand from the mid dims shape (bs :: seq_length)
        auto mask_reshape_node = mask_node->input(1)->node();
        auto qk_value = qk_node->input(0)->type()->cast<TensorType>();
        for (int i = 1; i < qk_value->dim().value() - 1; i++) {
          auto expand_check =
              toIValue(mask_reshape_node->inputs().at(i)).value().toInt();
          if (!(expand_check == 1)) {
            return false;
          }
        }

        return true;
      };

  auto filter_vit_mha =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        Node* node = match.anchor;
        // // Only support contiguous tensor for mask
        auto mask_value =
            node->input(0)->node()->input(1)->type()->cast<TensorType>();
        if (!utils::is_contiguous(mask_value)) {
          return false;
        }

        // Mask dim is the same as qk scores
        auto qk_node = node->input(0)->node();
        auto qk_value = qk_node->input(0)->type()->cast<TensorType>();
        if (!mask_value->dim().has_value() ||
            qk_value->dim().value() != mask_value->dim().value()) {
          return false;
        }

        // Only support the non-last-dimension broadcast
        auto not_last_dim_broadcast =
            mask_value->sizes()[mask_value->dim().value() - 1] != 1;
        if (!not_last_dim_broadcast) {
          return false;
        }
        return true;
      };

  SubgraphRewriter mha_fusion;
  SubgraphRewriter distil_mha_fusion;
  SubgraphRewriter maskedfill_softmax_fusion;
  SubgraphRewriter vit_mha_fusion;

  mha_fusion.RegisterRewritePattern(
      matmul_mul_add_softmax, matmul_mul_add_softmax_fusion);
  mha_fusion.RegisterRewritePattern(
      matmul_div_add_softmax, matmul_div_add_softmax_fusion);

  distil_mha_fusion.RegisterRewritePattern(
      matmul_div_maskedfill__softmax, matmul_div_maskedfill_softmax_fusion);
  distil_mha_fusion.RegisterRewritePattern(
      matmul_div_maskedfill_softmax, matmul_div_maskedfill_softmax_fusion);

  maskedfill_softmax_fusion.RegisterRewritePattern(
      expand_maskedfill__softmax, expand_maskedfill_softmax_fusion);
  maskedfill_softmax_fusion.RegisterRewritePattern(
      expand_maskedfill_softmax, expand_maskedfill_softmax_fusion);

  vit_mha_fusion.RegisterRewritePattern(
      matmul_mul_maskedfill_softmax, matmul_mul_maskedfill_softmax_fusion);
  vit_mha_fusion.RegisterRewritePattern(
      matmul_mul_maskedfill__softmax, matmul_mul_maskedfill_softmax_fusion);

  mha_fusion.runOnGraph(graph);
  distil_mha_fusion.runOnGraph(graph, {filter_common, filter_distil_mha});
  maskedfill_softmax_fusion.runOnGraph(
      graph, {filter_common, filter_distil_mha});
  vit_mha_fusion.runOnGraph(graph, {filter_common, filter_vit_mha});
}

// ipex::softmax shows better performance than aten::softmax, but compared with
// ipex::softmax_, it is slower.
// Like ipex::softmax_, we only do the replacement when the input
// is contiguous.
void replaceAtenSoftmaxWithIpexSoftmax(std::shared_ptr<Graph>& graph) {
  std::string aten_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = aten::softmax(%a, %dim, %half_to_float)
        return (%r) )";
  std::string ipex_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = ipex::softmax(%a, %dim, %half_to_float)
        return (%r) )";

  // Filter the unsupported case for inplace softmax
  auto filter_outplace =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        Node* node = match.anchor;
        // check if the input is contiguous, and skip if it is not
        auto input_value = node->input(0)->type()->cast<TensorType>();
        if (!utils::is_contiguous(input_value)) {
          return false;
        }

        return true;
      };

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

void replaceMergedEmbCatWithQmergedEmbCat(std::shared_ptr<Graph>& graph) {
  // graph->print(std::cout);
  std::vector<std::string> patterns;
  std::vector<std::string> replacements;
  std::string graph_common_head = R"(graph()";
  std::string graph_common_tail =
      R"(, %indices, %offsets, %qdense, %o_scale, %o_zp, %o_dtype):
  )";
  std::string list_construct_common_head =
      R"(%weights : Tensor[] = prim::ListConstruct()";
  std::string list_construct_common_tail = R"() )";
  std::string replacement_common_tail =
      R"(%out = ipex::qmerged_embeddingbag_cat(%weights, %indices,  %offsets, %qdense, %o_scale, %o_zp, %o_dtype) return (%out) )";
  std::string pattern_common_tail =
      R"(%dense=aten::dequantize(%qdense)  %out = torch_ipex::merged_embeddingbag_cat_forward(%weights, %indices, %offsets, %dense)  %qout = aten::quantize_per_tensor(%out, %o_scale, %o_zp, %o_dtype) return (%qout) )";

  for (auto* n : graph->block()->nodes()) {
    if (n->kind() ==
        Symbol::fromQualString("torch_ipex::merged_embeddingbag_cat_forward")) {
      size_t id = 0;
      auto weightslist = n->input(0)->node();

      bool is_quantized = std::any_of(
          weightslist->inputs().begin(),
          weightslist->inputs().end(),
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
      for (auto input : weightslist->inputs()) {
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

// When converting LSTM to int8 LSTM, IPEX will pre-hook the LSTM forward
// function to insert quant and dequant node. After converting the model, when
// entering the forward function, if the hidden state and cell state are empty,
// when generating dummy states, the graph will become:
//
// %quantized_input = aten::quantize_per_tensor(%input)
// %ret = aten::dequantize(%quantized_input)
// %max_batch_size : int = aten::size(%ret, %size_dim)
// ...
// %y, %hy, %cy = aten::lstm(%ret, ...)
//
// %ret is used by aten::size thus dequant-lstm cannot be fused.
// This pass will lift aten::size to be after quantize_per_tensor:
//
// %quantized_input = aten::quantize_per_tensor(%input)
// %max_batch_size : int = aten::size(%quantized_input, %size_dim)
//                         ^-- move size check before dequantize as it preserves
//                         shape
// %ret = aten::dequantize(%quantized_input)
// ...
// %y, %hy, %cy = aten::lstm(%ret, ...)
void preprocessSizeForQLstm(std::shared_ptr<Graph>& graph) {
  const static std::string op_list_construct_same_states = R"(
%hx.1 = aten::zeros(%sizes, %scalar_type, %layout, %device, %pin_memory) 
%state : Tensor[] = prim::ListConstruct(%hx.1, %hx.1) )";

  const static std::string op_list_construct_diff_states = R"(
%hx.1 = aten::zeros(%sizes, %scalar_type, %layout, %device, %pin_memory) 
%hx = aten::zeros(%sizes, %scalar_type, %layout, %device, %pin_memory)
%state : Tensor[] = prim::ListConstruct(%hx.1, %hx) )";

  std::vector<std::string> op_list_construct_sets = {
      op_list_construct_same_states, op_list_construct_diff_states};

  auto pattern = at::jit::CodeTemplate(R"(
     graph(%quantized_input, %size_dim, %ld, %hidden_size, %scalar_type, %layout, %device, %pin_memory, %weight, %has_biases, %num_layers, %dropout, %train, %bidirectional, %batch_first):
        %ret.3 = aten::dequantize(%quantized_input)
        %max_batch_size : int = aten::size(%ret.3, %size_dim)
        %ret.tensor : Tensor = prim::NumToTensor(%max_batch_size)
        %ret.int : int = aten::Int(%ret.tensor)
        %sizes : int[] = prim::ListConstruct(%ld, %ret.int, %hidden_size)
        ${op_list_construct}
        %res.1 : Tensor, %res.2 : Tensor, %res.3 : Tensor = aten::lstm(%ret.3, %state, %weight, %has_biases, %num_layers, %dropout, %train, %bidirectional, %batch_first)
        return (%res.1, %res.2, %res.3) )");

  auto replacement = at::jit::CodeTemplate(R"(
     graph(%quantized_input, %size_dim, %ld, %hidden_size, %scalar_type, %layout, %device, %pin_memory, %weight, %has_biases, %num_layers, %dropout, %train, %bidirectional, %batch_first):
        %max_batch_size : int = aten::size(%quantized_input, %size_dim)
        %ret.3 = aten::dequantize(%quantized_input)
        %ret.tensor : Tensor = prim::NumToTensor(%max_batch_size)
        %ret.int : int = aten::Int(%ret.tensor)
        %sizes : int[] = prim::ListConstruct(%ld, %ret.int, %hidden_size)
        ${op_list_construct}
        %res.1 : Tensor, %res.2 : Tensor, %res.3 : Tensor = aten::lstm(%ret.3, %state, %weight, %has_biases, %num_layers, %dropout, %train, %bidirectional, %batch_first)
        return (%res.1, %res.2, %res.3) )");

  for (auto const& it : op_list_construct_sets) {
    at::jit::TemplateEnv env;
    env.s("op_list_construct", it);

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        pattern.format(env), replacement.format(env));
    rewriter.runOnGraph(graph);
  }
}

void replaceLstmWithQLstm(std::shared_ptr<Graph>& graph) {
  std::vector<std::string> patterns;
  std::vector<std::string> replacements;

  for (auto* n : graph->block()->nodes()) {
    if (n->kind() == aten::lstm) {
      std::string weight_pattern = "";
      std::vector<std::string> ListConstruct;
      std::vector<std::string> header;

      size_t id = 0;
      auto weights_ListConstructNode = n->input(2)->node();

      bool maybe_quantized_lstm = std::any_of(
          weights_ListConstructNode->inputs().begin(),
          weights_ListConstructNode->inputs().end(),
          [](auto& v) {
            return v->node()->kind() == Symbol::aten("dequantize");
          });

      if (!maybe_quantized_lstm)
        return;

      for (auto input : weights_ListConstructNode->inputs()) {
        if (input->node()->kind() == Symbol::aten("dequantize")) {
          std::string dequant = "%dq_out_" + std::to_string(id) +
              " : Tensor = aten::dequantize(" + "%dq_in_" + std::to_string(id) +
              ")";
          weight_pattern.append(dequant);

          header.push_back("%dq_in_" + std::to_string(id));
          ListConstruct.push_back("%dq_out_" + std::to_string(id));
        } else {
          header.push_back("%bias_in_" + std::to_string(id));
          ListConstruct.push_back("%bias_in_" + std::to_string(id));
        }
        ++id;
      }

      std::string complete_header =
          "graph(%quantized_input, %h, %has_biases, %num_layers, %dropout_p, %train, %bidirectional, %batch_fist, %scale, %zp, %dtype," +
          c10::Join(", ", header) + R"(
              ): )";
      std::string complete_LC = "%weights = prim::ListConstruct(" +
          c10::Join(", ", ListConstruct) + ")";

      std::string QLstmPattern = complete_header + R"(
              %input : Tensor = aten::dequantize(%quantized_input) )" +
          weight_pattern + complete_LC + R"(
              %output, %hy, %cy = aten::lstm(%input, %h, %weights, %has_biases, %num_layers, %dropout_p, %train, %bidirectional, %batch_fist) 
              %quantized_output = aten::quantize_per_tensor(%output, %scale, %zp, %dtype)
              return (%quantized_output, %hy, %cy) )";

      std::string QLstmReplacement = complete_header + R"(
              %quantized_weights : Tensor[] = prim::ListConstruct( )" +
          c10::Join(", ", header) + R"(
              )
              %quantized_output, %hy, %cy = ipex::quantized_lstm(%quantized_input, %h, %quantized_weights, %has_biases, %num_layers, %dropout_p, %train, %bidirectional, %batch_fist, %scale, %zp, %dtype)
              return (%quantized_output, %hy, %cy) )";

      patterns.push_back(QLstmPattern);
      replacements.push_back(QLstmReplacement);
    }
  }

  SubgraphRewriter rewriter;
  for (size_t i = 0; i < patterns.size(); i++) {
    rewriter.RegisterRewritePattern(patterns[i], replacements[i]);
    rewriter.runOnGraph(graph);
  }
}

void replaceAddWithQAdd(std::shared_ptr<Graph>& graph) {
  std::string qadd = R"(
      graph(%add_a, %add_b, %alpha, %o_scale, %o_zp, %o_dtype):
        %r = quantized::add(%add_a, %add_b, %o_scale, %o_zp)
        return (%r) )";

  std::string add_with_quant_dequant = R"(
      graph(%add_a, %add_b, %alpha, %o_scale, %o_zp, %o_dtype):
        %dqadd_a = aten::dequantize(%add_a)
        %dqadd_b = aten::dequantize(%add_b)
        %r = aten::add(%dqadd_a, %dqadd_b, %alpha)
        %qout = aten::quantize_per_tensor(%r, %o_scale, %o_zp, %o_dtype)
        return (%qout) )";

  // fliter the unsupported case
  auto fusion_filter = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto alpha = match.values_map.at(vmap.at("alpha"));
    if (alpha->node()->kind() == prim::Constant) {
      auto alpha_value = toIValue(alpha).value();
      if (alpha_value.isDouble() && alpha_value.toDouble() == 1.0) {
        return true;
      } else if (alpha_value.isInt() && alpha_value.toInt() == 1) {
        return true;
      }
    }
    return false;
  };

  SubgraphRewriter rewriter_qadd;
  rewriter_qadd.RegisterRewritePattern(add_with_quant_dequant, qadd);
  rewriter_qadd.runOnGraph(graph, fusion_filter);
}

void fuseBmmAdd(std::shared_ptr<Graph>& graph) {
  std::array<std::string, 2> add_operators = {"add", "add_"};

  auto bmm_add_rstring_v1 = R"(
    graph(%input, %batch1, %batch2, %alpha):
        %x = aten::bmm(%batch1, %batch2)
        %res = aten::add(%x, %input, %alpha)
        return (%res))";
  std::string bmm_add_fused = R"(
    graph(%input, %batch1, %batch2, %alpha):
        %res = ipex::bmm_add(%input, %batch1, %batch2, %alpha)
        return (%res))";
  // fliter the unsupported case
  auto fusion_filter = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;

    auto batch1 = torch_ipex::jit::graph_rewrite_helper::getValue(
                      "batch1", match_vmap, vmap)
                      ->type()
                      ->cast<TensorType>();
    auto batch2 = torch_ipex::jit::graph_rewrite_helper::getValue(
                      "batch2", match_vmap, vmap)
                      ->type()
                      ->cast<TensorType>();
    if (batch1->dim() != batch2->dim()) {
      return false;
    }

    if (batch1->dim().value() < 3) {
      return false;
    }

    return true;
  };

  SubgraphRewriter rewriter_add_v1;
  rewriter_add_v1.RegisterRewritePattern(bmm_add_rstring_v1, bmm_add_fused);
  rewriter_add_v1.runOnGraph(graph, fusion_filter);
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
        %scale = aten::div(%weight, %u2)
        %u3 = aten::mul(%running_mean, %scale)
        %beta = aten::sub(%bias, %u3, %alpha)
        %b = ipex::concat_bn_relu(%input, %scale, %beta, %weight, %bias, %running_mean, %running_var, %training, %momentum, %eps, %cudnn_enabled, %dim)
        return (%b) )";

  auto fusion_filter = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    Node* node = match.anchor;
    const auto& match_vmap = match.values_map;
    // Check if the Concat Dimension is the channel
    auto dim_ = torch_ipex::jit::graph_rewrite_helper::getIValue(
                    "dim", match_vmap, vmap)
                    .value();
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
    int64_t list_length = 0;
    for (auto p : listConstruct->inputs()) {
      list_length++;
    }
    // Check if the Concat list is not empty
    TORCH_CHECK(list_length);

    auto tensor1 = listConstruct->input(0)->type()->cast<TensorType>();
    auto check_type_channelsize = [](c10::TensorType tensor) {
      return (
          (tensor.scalarType().value() == at::kFloat ||
           tensor.scalarType().value() == at::kBFloat16 ||
           tensor.scalarType().value() == at::kHalf) &&
          tensor.sizes()[1].value() % 16 == 0 &&
          utils::is_channelslast(tensor));
    };
    // Check if the dimension of the first tensor is either 4 or 5.
    // Check if the data type, the size of Channels, and the memory format are
    // float, mutiples of 16, and ChannelsLast(3d), respectively.
    if (!(tensor1->dim().value() == 4 || tensor1->dim().value() == 5) ||
        !check_type_channelsize(*tensor1)) {
      return false;
    }
    // Check the rest tensors
    for (int64_t i = 1; i < list_length; ++i) {
      auto tensori = listConstruct->input(i)->type()->cast<TensorType>();
      // Check dimension, data type, channel size and memory format
      if (!(tensor1->dim().value() == tensori->dim().value()) ||
          !check_type_channelsize(*tensori)) {
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
    // Check if the BN weights is fp32 datatype.
    auto bn_node = node->input(0)->node();
    if (bn_node->namedInput("weight")
            ->type()
            ->cast<TensorType>()
            ->scalarType()
            .value() != at::kFloat) {
      return false;
    }
    return true;
  };

  SubgraphRewriter rewriter_concatbnrelu;
  rewriter_concatbnrelu.RegisterRewritePattern(
      aten_concat_bn_relu, fused_concat_bn_relu);
  rewriter_concatbnrelu.runOnGraph(graph, fusion_filter);
}

void FuseLinearSwishCustomized(std::shared_ptr<Graph>& graph) {
  std::string linear_swish = R"(
      graph(%x, %weight, %bias):
        %_linear_res = aten::linear(%x, %weight, %bias)
        %_sigmod_res = aten::sigmoid(%_linear_res)
        %_mul_res2 = aten::mul(%_linear_res, %_sigmod_res)
        return (%_mul_res2) )";

  std::string linear_swish_fusion = R"(
      graph(%x, %weight, %bias):
        %_res = ipex::linear_swish_customized(%x, %weight, %bias)
        return (%_res) )";

  SubgraphRewriter ls_fusion;
  ls_fusion.RegisterRewritePattern(linear_swish, linear_swish_fusion);
  ls_fusion.runOnGraph(graph);
}

void FusePythonGELUWithAten(std::shared_ptr<Graph>& graph) {
  // from transformers common defined python gelu
  // ref:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L56
  std::string PythonGELUTanh_v1 = R"(
      graph(%x, %const_value1, %const_value2, %const_value3, %const_value4, %const_value5, %const_value6):
        %res_tmp1 = aten::mul(%x, %const_value1)
        %res_tmp2 = aten::pow(%x, %const_value2)
        %res_tmp2_ = aten::mul(%res_tmp2, %const_value3)
        %res_tmp3 = aten::add(%x, %res_tmp2_, %const_value4)
        %res_tmp4 = aten::mul(%res_tmp3, %const_value5)
        %res_tmp5 = aten::tanh(%res_tmp4)
        %res_tmp6 = aten::add(%res_tmp5, %const_value6, %const_value4)
        %res = aten::mul(%res_tmp1, %res_tmp6)
        return (%res) )";

  // from transformers defined python gelu in bloom model
  // ref:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L158
  std::string PythonGELUTanh_v2 = R"(
      graph(%x, %const_value1, %const_value2, %const_value3, %const_value4, %const_value5):
        %res_tmp1 = aten::mul(%x, %const_value1)
        %res_tmp2 = aten::mul(%x, %const_value2)
        %res_tmp3 = aten::mul(%x, %const_value3)
        %res_tmp3_ = aten::mul(%res_tmp3, %x)
        %res_tmp4 = aten::add(%res_tmp3_, %const_value4, %const_value4)
        %res_tmp5 = aten::mul(%res_tmp2, %res_tmp4)
        %res_tmp6 = aten::tanh(%res_tmp5)
        %res_tmp7 = aten::add(%res_tmp6, %const_value5, %const_value4)
        %res = aten::mul(%res_tmp1, %res_tmp7)
        return (%res) )";

  std::string AtenGELUTanh_argv1 = R"(
      graph(%x, %const_value1, %const_value2, %const_value3, %const_value4, %const_value5, %const_value6):
        %tanh_ : str = prim::Constant[value="tanh"]()
        %res = aten::gelu(%x, %tanh_)
        return (%res) )";

  std::string AtenGELUTanh_argv2 = R"(
      graph(%x, %const_value1, %const_value2, %const_value3, %const_value4, %const_value5):
        %tanh_ : str = prim::Constant[value="tanh"]()
        %res = aten::gelu(%x, %tanh_)
        return (%res) )";
  auto filter_v1 = [](const Match& match,
                      const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto const_value1 = toIValue(
        graph_rewrite_helper::getValue("const_value1", match_vmap, vmap));
    if (!const_value1.has_value() ||
        const_value1->toScalar().to<float>() != 0.5) {
      return false;
    }
    auto const_value2 = toIValue(
        graph_rewrite_helper::getValue("const_value2", match_vmap, vmap));
    if (!const_value2.has_value() ||
        const_value2->toScalar().to<float>() != 3.0) {
      return false;
    }
    auto const_value3 = toIValue(
        graph_rewrite_helper::getValue("const_value3", match_vmap, vmap));
    if (!const_value3.has_value() ||
        std::fabs(const_value3->toScalar().to<float>() - 0.044715) >
            0.0000001) {
      return false;
    }
    auto const_value4 = toIValue(
        graph_rewrite_helper::getValue("const_value4", match_vmap, vmap));
    if (!const_value4.has_value() || const_value4->toInt() != 1) {
      return false;
    }
    auto const_value5 = toIValue(
        graph_rewrite_helper::getValue("const_value5", match_vmap, vmap));
    if (!const_value5.has_value() ||
        std::fabs(const_value5->toScalar().to<float>() - 0.79788456) >
            0.0000001) {
      return false;
    }
    auto const_value6 = toIValue(
        graph_rewrite_helper::getValue("const_value6", match_vmap, vmap));
    if (!const_value6.has_value() ||
        const_value6->toScalar().to<float>() != 1.0) {
      return false;
    }
    return true;
  };

  auto filter_v2 = [](const Match& match,
                      const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto const_value1 = toIValue(
        graph_rewrite_helper::getValue("const_value1", match_vmap, vmap));
    if (!const_value1.has_value() ||
        const_value1->toScalar().to<float>() != 0.5) {
      return false;
    }
    auto const_value2 = toIValue(
        graph_rewrite_helper::getValue("const_value2", match_vmap, vmap));
    if (!const_value2.has_value() ||
        std::fabs(const_value2->toScalar().to<float>() - 0.79788456) >
            0.0000001) {
      return false;
    }
    auto const_value3 = toIValue(
        graph_rewrite_helper::getValue("const_value3", match_vmap, vmap));
    if (!const_value3.has_value() ||
        std::fabs(const_value3->toScalar().to<float>() - 0.044715) >
            0.0000001) {
      return false;
    }
    auto const_value4 = toIValue(
        graph_rewrite_helper::getValue("const_value4", match_vmap, vmap));
    if (!const_value4.has_value() || const_value4->toInt() != 1) {
      return false;
    }
    auto const_value5 = toIValue(
        graph_rewrite_helper::getValue("const_value5", match_vmap, vmap));
    if (!const_value5.has_value() ||
        const_value5->toScalar().to<float>() != 1.0) {
      return false;
    }
    return true;
  };

  SubgraphRewriter SingleGeluTanh_v1;
  SingleGeluTanh_v1.RegisterRewritePattern(
      PythonGELUTanh_v1, AtenGELUTanh_argv1);
  SingleGeluTanh_v1.runOnGraph(graph, filter_v1);
  SubgraphRewriter SingleGeluTanh_v2;
  SingleGeluTanh_v2.RegisterRewritePattern(
      PythonGELUTanh_v2, AtenGELUTanh_argv2);
  SingleGeluTanh_v2.runOnGraph(graph, filter_v2);
}

// This path will be removed after pytorch offical path is optimized well.
void replaceAtenMaxPool2dWithIpexMaxPool2d(std::shared_ptr<Graph>& graph) {
  std::string max_pool2d = R"(
      graph(%a, %kernel_size:int[], %stride:int[], %padding:int[], %dilation:int[], %ceil_mode:bool):
        %res = aten::max_pool2d(%a, %kernel_size, %stride, %padding, %dilation, %ceil_mode)
        return (%res) )";
  std::string ipex_max_pool2d = R"(
      graph(%a, %kernel_size:int[], %stride:int[], %padding:int[], %dilation:int[], %ceil_mode:bool):
        %res = ipex::max_pool2d(%a, %kernel_size, %stride, %padding, %dilation, %ceil_mode)
        return (%res) )";
  SubgraphRewriter rewriter_max_pool2d;
  rewriter_max_pool2d.RegisterRewritePattern(max_pool2d, ipex_max_pool2d);
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    auto pool_node = match.values_map.at(vmap.at("res"))->node();
    auto input = pool_node->inputs().at(0);
    if (!input->type()->cast<TensorType>()) {
      return false;
    }
    auto dtype_option = input->type()->cast<TensorType>()->scalarType();
    if (!dtype_option) {
      return false;
    }
    if (dtype_option.value() != c10::ScalarType::BFloat16 &&
        dtype_option.value() != c10::ScalarType::Float &&
        dtype_option.value() != c10::ScalarType::Half) {
      return false;
    }
    if (dtype_option.value() == c10::ScalarType::Half &&
        !torch_ipex::utils::onednn_has_fp16_type_support()) {
      return false;
    }
    return true;
  };
  rewriter_max_pool2d.runOnGraph(graph, filter);
}

void simplifyAllReduce(std::shared_ptr<Graph>& graph) {
  std::string all_reduce_v1 = R"(
    graph(%a, %weight, %out_features1, %out_features2, %b, %fc_in_weight, %fc_in_bias, %fc_out_weight, %fc_out_bias, %alpha, %idx, %no, %dtype, %zero):
      %r1 = torch_ipex::tpp_linear(%a, %weight, %out_features1)
      %r2 = deepspeed_comm::all_reduce(%r1)
      %r3 = torch_ipex::tpp_linear_gelu(%b, %fc_in_weight, %fc_in_bias, %out_features2)
      %r4 = aten::to(%r3, %idx, %no, %no, %dtype)
      %r5 = aten::contiguous(%r4, %zero)
      %r6 = torch_ipex::tpp_linear(%r5, %fc_out_weight, %out_features1)
      %r7 = deepspeed_comm::all_reduce(%r6)
      %r8 = aten::add_(%r7, %fc_out_bias, %alpha)
      %r = aten::add(%r2, %r8, %alpha)
      return (%r) )";
  std::string all_reduce_repl_v1 = R"(
    graph(%a, %weight, %out_features1, %out_features2, %b, %fc_in_weight, %fc_in_bias, %fc_out_weight, %fc_out_bias, %alpha, %idx, %no, %dtype, %zero):
      %r1 = torch_ipex::tpp_linear(%a, %weight, %out_features1)
      %r2 = torch_ipex::tpp_linear_gelu(%b, %fc_in_weight, %fc_in_bias, %out_features2)
      %r3 = aten::to(%r2, %idx, %no, %no, %dtype)
      %r4 = aten::contiguous(%r3, %zero)
      %r5 = torch_ipex::tpp_linear(%r4, %fc_out_weight, %out_features1)
      %r6 = aten::add(%r1, %r5, %alpha)
      %r7 = deepspeed_comm::all_reduce(%r6)
      %r = aten::add_(%r7, %fc_out_bias, %alpha)
      return (%r) )";

  std::string all_reduce_v2 = R"(
    graph(%a, %weight, %b, %fc_in_weight, %fc_in_bias, %fc_out_weight, %fc_out_bias, %alpha):
      %r1 = ipex_prepack::linear_run(%a, %weight)
      %r2 = deepspeed_comm::all_reduce(%r1)
      %r3 = ipex_prepack::linear_gelu_run(%b, %fc_in_weight, %fc_in_bias)
      %r4 = ipex_prepack::linear_run(%r3, %fc_out_weight)
      %r5 = deepspeed_comm::all_reduce(%r4)
      %r6 = aten::add_(%r5, %fc_out_bias, %alpha)
      %r = aten::add(%r2, %r6, %alpha)
      return (%r) )";
  std::string all_reduce_repl_v2 = R"(
    graph(%a, %weight, %b, %fc_in_weight, %fc_in_bias, %fc_out_weight, %fc_out_bias, %alpha):
      %r1 = ipex_prepack::linear_run(%a, %weight)
      %r2 = ipex_prepack::linear_gelu_run(%b, %fc_in_weight, %fc_in_bias)
      %r3 = ipex_prepack::linear_run(%r2, %fc_out_weight)
      %r4 = aten::add(%r1, %r3, %alpha)
      %r5 = deepspeed_comm::all_reduce(%r4)
      %r = aten::add_(%r5, %fc_out_bias, %alpha)
      return (%r) )";

  std::string all_reduce_v3 = R"(
    graph(%a, %weight, %out_features1, %out_features2, %b, %fc_in_weight, %fc_in_bias, %fc_out_weight, %fc_out_bias, %alpha, %idx, %no, %dtype, %zero):
      %r1 = torch_ipex::tpp_linear(%a, %weight, %out_features1)
      %r2 = torch_ipex::inference_all_reduce_add(%r1)
      %r2_1 = aten::to(%b, %idx, %no, %no, %dtype)
      %r2_2 = aten::contiguous(%r2_1, %zero)
      %r3 = torch_ipex::tpp_linear_gelu(%r2_2, %fc_in_weight, %fc_in_bias, %out_features2)
      %r4 = aten::to(%r3, %idx, %no, %no, %dtype)
      %r5 = aten::contiguous(%r4, %zero)
      %r6 = torch_ipex::tpp_linear_bias(%r5, %fc_out_weight, %fc_out_bias, %out_features1)
      %r7 = torch_ipex::inference_all_reduce_add(%r6)
      %r = aten::add(%r2, %r7, %alpha)
      return (%r) )";
  std::string all_reduce_repl_v3 = R"(
    graph(%a, %weight, %out_features1, %out_features2, %b, %fc_in_weight, %fc_in_bias, %fc_out_weight, %fc_out_bias, %alpha, %idx, %no, %dtype, %zero):
      %r1 = torch_ipex::tpp_linear(%a, %weight, %out_features1)
      %r2_1 = aten::to(%b, %idx, %no, %no, %dtype)
      %r2_2 = aten::contiguous(%r2_1, %zero)
      %r2 = torch_ipex::tpp_linear_gelu(%r2_2, %fc_in_weight, %fc_in_bias, %out_features2)
      %r3 = aten::to(%r2, %idx, %no, %no, %dtype)
      %r4 = aten::contiguous(%r3, %zero)
      %scale: float = prim::Constant[value=1.0]()
      %r5 = torch_ipex::tpp_linear_add(%r4, %r1, %fc_out_weight, %fc_out_bias, %scale, %out_features1)
      %r6 = torch_ipex::inference_all_reduce_add(%r5)
      return (%r6) )";

  SubgraphRewriter rewriter_v1, rewriter_v2, rewriter_v3;
  rewriter_v1.RegisterRewritePattern(all_reduce_v1, all_reduce_repl_v1);
  rewriter_v2.RegisterRewritePattern(all_reduce_v2, all_reduce_repl_v2);
  rewriter_v3.RegisterRewritePattern(all_reduce_v3, all_reduce_repl_v3);
  rewriter_v1.runOnGraph(graph);
  rewriter_v2.runOnGraph(graph);
  rewriter_v3.runOnGraph(graph);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
