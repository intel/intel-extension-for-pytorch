#include "graph_rewrite.h"
#include "graph_rewrite_helper.h"
#include "graph_rewrite_utils.h"

#include <ATen/code_template.h>

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;
using namespace torch::jit;

auto transfree_mha_filter = [](const Match& match,
                               const std::unordered_map<std::string, Value*>&
                                   vmap) {
  const auto& match_vmap = match.values_map;
  auto permute_sizes =
      toIValue(graph_rewrite_helper::getValue("permute", match_vmap, vmap))
          ->toIntVector();
  auto qkv =
      torch_ipex::jit::graph_rewrite_helper::getValue("qkv", match_vmap, vmap)
          ->type()
          ->cast<TensorType>();
  auto trans_a =
      toIValue(graph_rewrite_helper::getValue("trans_a", match_vmap, vmap))
          ->toInt();
  auto trans_b =
      toIValue(graph_rewrite_helper::getValue("trans_b", match_vmap, vmap))
          ->toInt();
  auto slice_dim =
      toIValue(graph_rewrite_helper::getValue("slice_neg1", match_vmap, vmap))
          ->toInt();
  auto slice_idx1 =
      toIValue(graph_rewrite_helper::getValue("slice_idx1", match_vmap, vmap))
          ->toInt();
  auto slice_idx2 =
      toIValue(graph_rewrite_helper::getValue("slice_idx2", match_vmap, vmap))
          ->toInt();
  auto slice_idx3 =
      toIValue(graph_rewrite_helper::getValue("slice_idx3", match_vmap, vmap))
          ->toInt();
  auto slice_idx4 =
      toIValue(graph_rewrite_helper::getValue("slice_idx4", match_vmap, vmap))
          ->toInt();
  std::vector<int64_t> permute_ref = {0, 2, 1, 3};
  if (permute_sizes != permute_ref || !(trans_a == -1 && trans_b == -2) ||
      slice_dim != -1 ||
      (slice_idx1 - slice_idx2) != (slice_idx2 - slice_idx3) ||
      (slice_idx2 - slice_idx3) != (slice_idx3 - slice_idx4) ||
      qkv->scalarType().value() == at::kFloat) {
    return false;
  }
  return true;
};

auto transfree_distil_mha_filter =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      const auto& match_vmap = match.values_map;
      auto bmm1 = graph_rewrite_helper::getValue("bmm1", match_vmap, vmap)
                      ->type()
                      ->cast<TensorType>();
      auto trans_a =
          toIValue(graph_rewrite_helper::getValue("trans_a", match_vmap, vmap))
              ->toInt();
      auto trans_b =
          toIValue(graph_rewrite_helper::getValue("trans_b", match_vmap, vmap))
              ->toInt();
      auto trans_c =
          toIValue(graph_rewrite_helper::getValue("trans_c", match_vmap, vmap))
              ->toInt();
      auto qkv = torch_ipex::jit::graph_rewrite_helper::getValue(
                     "qkv", match_vmap, vmap)
                     ->type()
                     ->cast<TensorType>();
      auto slice_dim = toIValue(graph_rewrite_helper::getValue(
                                    "slice_neg1", match_vmap, vmap))
                           ->toInt();
      auto slice_idx1 = toIValue(graph_rewrite_helper::getValue(
                                     "slice_idx1", match_vmap, vmap))
                            ->toInt();
      auto slice_idx2 = toIValue(graph_rewrite_helper::getValue(
                                     "slice_idx2", match_vmap, vmap))
                            ->toInt();
      auto slice_idx3 = toIValue(graph_rewrite_helper::getValue(
                                     "slice_idx3", match_vmap, vmap))
                            ->toInt();
      auto slice_idx4 = toIValue(graph_rewrite_helper::getValue(
                                     "slice_idx4", match_vmap, vmap))
                            ->toInt();
      if (bmm1->dim().value() != 4 ||
          !(trans_a == 1 && trans_b == 2 && trans_c == 3) || slice_dim != -1 ||
          (slice_idx1 - slice_idx2) != (slice_idx2 - slice_idx3) ||
          (slice_idx2 - slice_idx3) != (slice_idx3 - slice_idx4) ||
          qkv->scalarType().value() == at::kFloat) {
        return false;
      }
      return true;
    };

auto vit_mha_fusion_filter =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      const auto& match_vmap = match.values_map;
      auto permute_sizes =
          toIValue(torch_ipex::jit::graph_rewrite_helper::getValue(
                       "qkv_permute", match_vmap, vmap))
              ->toIntVector();
      auto trans_a =
          toIValue(graph_rewrite_helper::getValue("trans_a", match_vmap, vmap))
              ->toInt();
      auto trans_b =
          toIValue(graph_rewrite_helper::getValue("trans_b", match_vmap, vmap))
              ->toInt();
      auto qkv_div = toIValue(torch_ipex::jit::graph_rewrite_helper::getValue(
                                  "qkv_div", match_vmap, vmap))
                         .value();
      auto q_select = toIValue(torch_ipex::jit::graph_rewrite_helper::getValue(
                                   "select_dim", match_vmap, vmap))
                          .value();
      auto k_select = toIValue(torch_ipex::jit::graph_rewrite_helper::getValue(
                                   "key_select", match_vmap, vmap))
                          .value();
      auto v_select = toIValue(torch_ipex::jit::graph_rewrite_helper::getValue(
                                   "value_select", match_vmap, vmap))
                          .value();
      auto qkv = torch_ipex::jit::graph_rewrite_helper::getValue(
                     "qkv", match_vmap, vmap)
                     ->type()
                     ->cast<TensorType>();
      std::vector<int64_t> permute_ref = {2, 0, 3, 1, 4};
      if (permute_sizes != permute_ref || qkv_div != 3 || q_select != 0 ||
          k_select != 1 || v_select != 2 ||
          !((trans_a == -2 && trans_b == -1) ||
            (trans_a == -1 && trans_b == -2)) ||
          qkv->scalarType().value() == at::kFloat) {
        return false;
      }
      return true;
    };

void FusedTransFreeMha(std::shared_ptr<Graph>& graph) {
  std::string bert_mha_graph = R"(
      graph(%qkv: Tensor, %slice_idx1: int, %slice_idx2: int, %slice_idx3: int, %slice_idx4: int, %slice_pos1: int, %slice_neg1: int, %one_p: int, %zero: int, %num_head: int, %head_dim: int, %permute: int[], %trans_a: int, %trans_b: int, %relative_qk: Tensor, %scale: int, %dtype): )";

  std::string mha_slice = R"(
        %query = aten::slice(%qkv, %slice_neg1, %slice_idx1, %slice_idx2, %slice_pos1)
        %key = aten::slice(%qkv, %slice_neg1, %slice_idx2, %slice_idx3, %slice_pos1)
        %value = aten::slice(%qkv, %slice_neg1, %slice_idx3, %slice_idx4, %slice_pos1) )";

  std::string bert_mha_main = R"(
        %query_size1 = aten::size(%query, %zero)
        %query_size2 = aten::size(%query, %one_p)
        %query_size = prim::ListConstruct(%query_size1, %query_size2, %num_head, %head_dim)
        %query_1 = aten::view(%query, %query_size)
        %query_layer = aten::permute(%query_1, %permute)
        %key_size1 = aten::size(%key, %zero)
        %key_size2 = aten::size(%key, %one_p)
        %key_size = prim::ListConstruct(%key_size1, %key_size2, %num_head, %head_dim)
        %key_1 = aten::view(%key, %key_size)
        %key_2 = aten::permute(%key_1, %permute)
        %key_layer = aten::transpose(%key_2, %trans_a, %trans_b)
        %bmm1 = ipex::mha_scores_calc(%query_layer, %key_layer, %relative_qk, %one_p, %scale, %trans_a, %dtype)
        %value_size1 = aten::size(%value, %zero)
        %value_size2 = aten::size(%value, %one_p)
        %value_size = prim::ListConstruct(%value_size1, %value_size2, %num_head, %head_dim)
        %value_1 = aten::view(%value, %value_size)
        %value_layer = aten::permute(%value_1, %permute)
        %bmm2 = aten::matmul(%bmm1, %value_layer)
        %context_layer1  = aten::permute(%bmm2, %permute)
        %context_layer = aten::contiguous(%context_layer1, %zero)
        return (%context_layer) )";

  std::string transfree_bert_mha = R"(
        %output = ipex::transfree_mha(%qkv, %relative_qk, %one_p, %scale, %trans_a, %dtype, %num_head, %head_dim)
        return (%output) )";

  std::string distil_mha_graph = R"(
      graph(%qkv: Tensor, %slice_idx1: int, %slice_idx2: int, %slice_idx3: int, %slice_idx4: int, %slice_pos1: int, %slice_neg1: int, %bs: int, %one_n: int, %num_head: int, %head_dim: int, %trans_a: int, %trans_b: int, %mask_qk_reshp: int[], %mask: Tensor, %zero: int, %trans_c:int, %fill:float, %dim_per_head:float): )";

  std::string distil_mha_main = R"(
        %view_size = prim::ListConstruct(%bs, %one_n, %num_head, %head_dim)
        %query_1 = aten::view(%query, %view_size)
        %query_layer = aten::transpose(%query_1, %trans_a, %trans_b)
        %key_1 = aten::view(%key, %view_size)
        %key_2 = aten::transpose(%key_1, %trans_a, %trans_b)
        %key_layer = aten::transpose(%key_2, %trans_b, %trans_c)
        %value_1 = aten::view(%value, %view_size)
        %value_layer = aten::transpose(%value_1, %trans_a, %trans_b)
        %bmm1 = ipex::distil_mha_scores_calc(%query_layer, %key_layer, %mask, %mask_qk_reshp, %fill, %dim_per_head)
        %bmm2 = aten::matmul(%bmm1, %value_layer)
        %context_layer1  = aten::transpose(%bmm2, %trans_a, %trans_b)
        %context_layer = aten::contiguous(%context_layer1, %zero)
        return (%context_layer) )";

  std::string transfree_distil_mha = R"(
        %output = ipex::transfree_distil_mha(%qkv, %mask, %mask_qk_reshp, %fill, %dim_per_head, %num_head, %head_dim)
        return (%output) )";

  auto bert_mha_pattern = bert_mha_graph + mha_slice + bert_mha_main;
  auto transfree_bert_mha_pattern = bert_mha_graph + transfree_bert_mha;
  auto distil_mha_pattern = distil_mha_graph + mha_slice + distil_mha_main;
  auto transfree_distil_mha_pattern = distil_mha_graph + transfree_distil_mha;

  SubgraphRewriter bert_mha_fusion, distil_mha_fusion;
  bert_mha_fusion.RegisterRewritePattern(
      bert_mha_pattern, transfree_bert_mha_pattern);
  distil_mha_fusion.RegisterRewritePattern(
      distil_mha_pattern, transfree_distil_mha_pattern);
  bert_mha_fusion.runOnGraph(graph, transfree_mha_filter);
  distil_mha_fusion.runOnGraph(graph, transfree_distil_mha_filter);

  std::string vit_mha_pattern = R"(
      graph(%bs: int, %seq: int, %qkv_div: int, %num_head: int, %head_size: int, %qkv: Tensor, %qkv_permute: int[], %select_dim: int, %key_select: int, %value_select: int, %trans_a: int, %trans_b: int, %scale, %dtype):
        %qkv_size = prim::ListConstruct(%bs, %seq, %qkv_div, %num_head, %head_size)
        %qkv1 = aten::reshape(%qkv, %qkv_size)
        %qkv2 = aten::permute(%qkv1, %qkv_permute)
        %query = aten::select(%qkv2, %select_dim, %select_dim)
        %key_ = aten::select(%qkv2, %select_dim, %key_select)
        %value = aten::select(%qkv2, %select_dim, %value_select)
        %key = aten::transpose(%key_, %trans_a, %trans_b)
        %bmm1 = ipex::matmul_div(%query, %key, %scale)
        %smx = ipex::softmax(%bmm1, %trans_b, %dtype)
        %bmm2 = aten::matmul(%smx, %value)
        %context_layer = aten::transpose(%bmm2, %key_select, %value_select)
        return (%context_layer) )";
  std::string transfree_vit_mha_pattern = R"(
      graph(%bs: int, %seq: int, %qkv_div: int, %num_head: int, %head_size: int, %qkv: Tensor, %qkv_permute: int[], %select_dim: int, %key_select: int, %value_select: int, %trans_a: int, %trans_b: int, %scale, %dtype):
        %output = ipex::transfree_vit_mha(%qkv, %scale, %trans_b, %dtype, %num_head, %head_size)
        return (%output) )";

  SubgraphRewriter vit_mha_fusion;
  vit_mha_fusion.RegisterRewritePattern(
      vit_mha_pattern, transfree_vit_mha_pattern);
  vit_mha_fusion.runOnGraph(graph, vit_mha_fusion_filter);
}
} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
