#include "graph_rewrite.h"
#include "graph_rewrite_helper.h"
#include "graph_rewrite_utils.h"

#include <ATen/code_template.h>

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;
using namespace torch::jit;
auto bert_flash_mha_filter =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      const auto& match_vmap = match.values_map;
      auto permute_sizes =
          toIValue(graph_rewrite_helper::getValue("permute", match_vmap, vmap))
              ->toIntVector();
      auto qkv = torch_ipex::jit::graph_rewrite_helper::getValue(
                     "qkv", match_vmap, vmap)
                     ->type()
                     ->cast<TensorType>();
      auto trans_a =
          toIValue(graph_rewrite_helper::getValue("trans_a", match_vmap, vmap))
              ->toInt();
      auto trans_b =
          toIValue(graph_rewrite_helper::getValue("trans_b", match_vmap, vmap))
              ->toInt();
      std::vector<int64_t> permute_ref = {0, 2, 1, 3};
      if (permute_sizes != permute_ref || !(trans_a == -1 && trans_b == -2) ||
          qkv->scalarType().value() != at::kBFloat16) {
        return false;
      }
      // Checking the dtype as None
      auto dtype_value = torch_ipex::jit::graph_rewrite_helper::getIValue(
          "dtype", match_vmap, vmap);
      if (!dtype_value.has_value() || !dtype_value.value().isNone()) {
        return false;
      }
      auto alpha =
          toIValue(graph_rewrite_helper::getValue("one_p", match_vmap, vmap))
              ->toScalar()
              .to<float>();
      if (alpha != 1.0f) {
        return false;
      }
      return true;
    };

auto sd_flash_mha_filter_v1 = [](const Match& match,
                                 const std::unordered_map<std::string, Value*>&
                                     vmap) {
  const auto& match_vmap = match.values_map;
  auto split_idx =
      toIValue(graph_rewrite_helper::getValue("split_idx", match_vmap, vmap))
          ->toIntVector();
  auto permute_sizes =
      toIValue(graph_rewrite_helper::getValue("permutelist", match_vmap, vmap))
          ->toIntVector();
  auto qkv =
      torch_ipex::jit::graph_rewrite_helper::getValue("qkv", match_vmap, vmap)
          ->type()
          ->cast<TensorType>();
  auto zero = toIValue(graph_rewrite_helper::getValue("zero", match_vmap, vmap))
                  ->toInt();
  auto neg_one =
      toIValue(graph_rewrite_helper::getValue("neg_one", match_vmap, vmap))
          ->toInt();
  auto neg_two =
      toIValue(graph_rewrite_helper::getValue("neg_two", match_vmap, vmap))
          ->toInt();
  auto one = toIValue(graph_rewrite_helper::getValue("one", match_vmap, vmap))
                 ->toInt();
  auto two = toIValue(graph_rewrite_helper::getValue("two", match_vmap, vmap))
                 ->toInt();
  std::vector<int64_t> permute_ref = {0, 2, 1, 3};
  if (permute_sizes != permute_ref ||
      !(zero == 0 && neg_one == -1 && neg_two == -2 && one == 1 && two == 2) ||
      qkv->scalarType().value() != at::kBFloat16 || split_idx.size() != 3 ||
      split_idx[0] != split_idx[1] || split_idx[0] != split_idx[2]) {
    return false;
  }
  return true;
};

auto sd_flash_mha_filter_v2 = [](const Match& match,
                                 const std::unordered_map<std::string, Value*>&
                                     vmap) {
  const auto& match_vmap = match.values_map;
  auto permute_sizes =
      toIValue(graph_rewrite_helper::getValue("permutelist", match_vmap, vmap))
          ->toIntVector();
  auto query0 = torch_ipex::jit::graph_rewrite_helper::getValue(
                    "query0", match_vmap, vmap)
                    ->type()
                    ->cast<TensorType>();
  auto zero = toIValue(graph_rewrite_helper::getValue("zero", match_vmap, vmap))
                  ->toInt();
  auto neg_one =
      toIValue(graph_rewrite_helper::getValue("neg_one", match_vmap, vmap))
          ->toInt();
  auto neg_two =
      toIValue(graph_rewrite_helper::getValue("neg_two", match_vmap, vmap))
          ->toInt();
  auto one = toIValue(graph_rewrite_helper::getValue("one", match_vmap, vmap))
                 ->toInt();
  auto two = toIValue(graph_rewrite_helper::getValue("two", match_vmap, vmap))
                 ->toInt();
  std::vector<int64_t> permute_ref = {0, 2, 1, 3};
  if (permute_sizes != permute_ref ||
      !(zero == 0 && neg_one == -1 && neg_two == -2 && one == 1 && two == 2) ||
      query0->scalarType().value() != at::kBFloat16) {
    return false;
  }
  return true;
};

auto sd_flash_mha_filter_v3 =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      const auto& match_vmap = match.values_map;
      auto split_idx = toIValue(graph_rewrite_helper::getValue(
                                    "split_idx", match_vmap, vmap))
                           ->toIntVector();
      auto one =
          toIValue(graph_rewrite_helper::getValue("one", match_vmap, vmap))
              ->toInt();
      auto two =
          toIValue(graph_rewrite_helper::getValue("two", match_vmap, vmap))
              ->toInt();
      auto neg_one =
          toIValue(graph_rewrite_helper::getValue("neg_one", match_vmap, vmap))
              ->toInt();
      auto qkv = torch_ipex::jit::graph_rewrite_helper::getValue(
                     "qkv", match_vmap, vmap)
                     ->type()
                     ->cast<TensorType>();
      if (!(one == 1 && two == 2 && neg_one == -1) ||
          qkv->scalarType().value() != at::kBFloat16 || split_idx.size() != 3 ||
          split_idx[0] != split_idx[1] || split_idx[0] != split_idx[2]) {
        return false;
      }
      return true;
    };

auto sd_flash_mha_filter_v4 =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      const auto& match_vmap = match.values_map;
      auto one =
          toIValue(graph_rewrite_helper::getValue("one", match_vmap, vmap))
              ->toInt();
      auto two =
          toIValue(graph_rewrite_helper::getValue("two", match_vmap, vmap))
              ->toInt();
      auto neg_one =
          toIValue(graph_rewrite_helper::getValue("neg_one", match_vmap, vmap))
              ->toInt();
      auto query0 = torch_ipex::jit::graph_rewrite_helper::getValue(
                        "query0", match_vmap, vmap)
                        ->type()
                        ->cast<TensorType>();
      if (!(one == 1 && two == 2 && neg_one == -1) ||
          query0->scalarType().value() != at::kBFloat16) {
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
          qkv->scalarType().value() != at::kBFloat16) {
        return false;
      }
      return true;
    };

auto transfree_bmm_filter =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      Node* node = match.anchor;
      const auto& match_vmap = match.values_map;

      auto batch1 = node->input(0)->type()->cast<TensorType>();

      auto batch2 = node->input(1)->type()->cast<TensorType>();

      if (!batch1->dim().has_value() || !batch2->dim().has_value() ||
          !batch1->scalarType().has_value() ||
          !batch2->scalarType().has_value()) {
        return false;
      }

      if (batch1->dim() != batch2->dim() || batch1->dim().value() < 3 ||
          batch1->sizes()[batch1->dim().value() - 1].value() !=
              batch2->sizes()[batch2->dim().value() - 2].value()) {
        return false;
      }

      for (int64_t i = 0; i < batch1->dim().value() - 2; ++i) {
        if (batch1->sizes()[i].value() != batch2->sizes()[i].value()) {
          return false;
        }
      }

      return true;
    };

auto bmm_outtrans_filter_v1 =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      Node* node = match.anchor;
      const auto& match_vmap = match.values_map;
      if (!toIValue(node->input(1)).has_value()) {
        return false;
      }
      auto permute_sizes = toIValue(node->input(1))->toIntVector();
      std::vector<int64_t> permute_ref = {0, 2, 1, 3};
      if (permute_sizes != permute_ref) {
        return false;
      }
      return true;
    };

auto bmm_outtrans_filter_v2 =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      Node* node = match.anchor;
      const auto& match_vmap = match.values_map;
      auto bmm1 = node->input(0)->node()->input(0)->type()->cast<TensorType>();
      if (!toIValue(node->input(1)).has_value() ||
          !toIValue(node->input(2)).has_value() || !bmm1->dim().has_value()) {
        return false;
      }
      auto trans_a = toIValue(node->input(1)).value();
      auto trans_b = toIValue(node->input(2)).value();
      if (bmm1->dim().value() != 4 || !(trans_a == 1 && trans_b == 2)) {
        return false;
      }
      return true;
    };

auto split_replace_filter =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      const auto& match_vmap = match.values_map;
      auto to_split =
          graph_rewrite_helper::getValue("to_split", match_vmap, vmap)
              ->type()
              ->cast<TensorType>();
      if (!to_split->scalarType().has_value() ||
          to_split->scalarType().value() != at::kBFloat16)
        return false;
      auto dim =
          toIValue(graph_rewrite_helper::getValue("dim", match_vmap, vmap))
              ->toInt();
      if (dim != -1)
        return false;
      return true;
    };

// aten::matmul - always applies contiguous to the input tensors
// ipex::matmul - allows non-contiguous input tensors with the conditions:
// 1. tensor1.dim1 == tensor2.dim2
// 2. tensor.dim >= 3
// 3. tensor.stride(-1) == 1 || tensor.stride(-2) == 1
// 4. tensor.sizes[0:dim-2] == tensor.sizes[0:dim-2]
// If the above conditions are satisfied, the ipex::matmul will use the
// non-contiguous input tensors for the computation to save unnecessary
// memory copies.
// ipex::matmul_outtrans - post fuses a specific transpose OP for MHA if
// the tensor.dim == 4 and the transpose indices are (1, 2) or
// the permute list is [0, 2, 1, 3].
void FusedTransFreeMha(std::shared_ptr<Graph>& graph) {
  // ViT MHA Fusion is using DNNL Transpose-free Matmul primitive tags.
  // Todo: Transfer to the Flash Attention.
  std::string aten_split_pattern = R"(
      graph(%to_split: Tensor, %split_list: int[], %dim: int):
        %out = aten::split_with_sizes(%to_split, %split_list, %dim)
        return (%out) )";

  std::string ipex_split_pattern = R"(
      graph(%to_split: Tensor, %split_list: int[], %dim: int):
        %out = ipex::split_tensor(%to_split, %split_list)
        return (%out) )";

  SubgraphRewriter split_replacer;
  split_replacer.RegisterRewritePattern(aten_split_pattern, ipex_split_pattern);
  split_replacer.runOnGraph(graph, split_replace_filter);

  std::string vit_mha_pattern = R"(
      graph(%bs: int, %seq: int, %qkv_div: int, %num_head: int, %head_size: int, %qkv: Tensor, %qkv_permute: int[], %select_dim: int, %key_select: int, %value_select: int, %trans_a: int, %trans_b: int, %scale, %dtype):
        %qkv_size = prim::ListConstruct(%bs, %seq, %qkv_div, %num_head, %head_size)
        %qkv1 = aten::reshape(%qkv, %qkv_size)
        %qkv2 = aten::permute(%qkv1, %qkv_permute)
        %query = aten::select(%qkv2, %select_dim, %select_dim)
        %key_ = aten::select(%qkv2, %select_dim, %key_select)
        %value = aten::select(%qkv2, %select_dim, %value_select)
        %key = aten::transpose(%key_, %trans_a, %trans_b)
        %bmm1 = ipex::matmul_mul(%query, %key, %scale)
        %smx = ipex::softmax(%bmm1, %trans_b, %dtype)
        %bmm2 = aten::matmul(%smx, %value)
        %context_layer = aten::transpose(%bmm2, %key_select, %value_select)
        return (%context_layer) )";

  std::string bert_flash_mha = R"(
        %output = ipex::bert_flash_mha(%qkv, %relative_qk, %one_p, %scale, %trans_a, %dtype, %num_head, %head_dim)
        return (%output) )";

  std::string transfree_vit_mha_pattern = R"(
      graph(%bs: int, %seq: int, %qkv_div: int, %num_head: int, %head_size: int, %qkv: Tensor, %qkv_permute: int[], %select_dim: int, %key_select: int, %value_select: int, %trans_a: int, %trans_b: int, %scale, %dtype):
        %output = ipex::transfree_vit_mha(%qkv, %scale, %trans_b, %dtype, %num_head, %head_size)
        return (%output) )";

  SubgraphRewriter vit_mha_fusion;
  vit_mha_fusion.RegisterRewritePattern(
      vit_mha_pattern, transfree_vit_mha_pattern);
  vit_mha_fusion.runOnGraph(graph, vit_mha_fusion_filter);

  // BERT and Stable-Diffusion MHA fusions are using the Flash Attention
  // Optimization scheme. Todo: Add DistilBERT MHA fusion.
  std::string bert_mha_graph = R"(
      graph(%qkv: Tensor, %split_idx: int[], %one_p: int, %zero: int, %num_head: int, %head_dim: int, %permute: int[], %trans_a: int, %trans_b: int, %relative_qk: Tensor, %scale: int, %dtype): )";

  std::string mha_slice = R"(
        %qkv_list = ipex::split_tensor(%qkv, %split_idx)
        %query, %key, %value = prim::ListUnpack(%qkv_list) )";

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

  auto bert_mha_pattern = bert_mha_graph + mha_slice + bert_mha_main;
  auto bert_flash_mha_pattern = bert_mha_graph + bert_flash_mha;
  SubgraphRewriter bert_mha_fusion;
  bert_mha_fusion.RegisterRewritePattern(
      bert_mha_pattern, bert_flash_mha_pattern);
  bert_mha_fusion.runOnGraph(graph, bert_flash_mha_filter);

  /**
   * Diffusers 0.12.1 uses aten::baddbmm / softmax / bmm to formulate
   * the MHA structure, while Diffusers 0.13.0 uses
   * aten::scaled_dot_product_attention to calculate MHA.
   * 0.12.1 uses the first ipex::sd_flash_mha kernels, and
   * 0.13.0 uses the latter two. Since 0.12.1 is widely
   * used as of 2023/02/20, it is better to keep both graph patterns.
   */
  std::string sd_mha_graph_v1 = R"(
      graph(%qkv: Tensor, %split_idx: int[], %zero, %neg_one, %neg_two, %one, %two, %idx, %scale: float, %no, %device, %dtype, %headsize, %num_head, %permutelist): )";

  std::string sd_mha_graph_v2 = R"(
      graph(%query0: Tensor, %key0: Tensor, %value0: Tensor, %zero, %neg_one, %neg_two, %one, %two, %idx, %scale: float, %no, %device, %dtype, %headsize, %num_head, %permutelist): )";

  std::string sd_mha_graph_v3 = R"(
      graph(%qkv: Tensor, %split_idx: int[], %one, %two, %neg_one, %num_head, %batchsize, %headsize, %hiddensize, %dropout, %idx, %no, %dtype, %scale): )";

  std::string sd_mha_graph_v4 = R"(
      graph(%query0: Tensor, %key0: Tensor, %value0: Tensor, %one, %two, %neg_one, %num_head, %batchsize, %headsize, %hiddensize, %dropout, %idx, %no, %dtype, %scale): )";

  std::string sd_qkv_split = R"(
        %qkv_list = ipex::split_tensor(%qkv, %split_idx)
        %query0, %key0, %value0 = prim::ListUnpack(%qkv_list) )";

  std::string sd_mha_query = R"(
        %query1 = aten::size(%query0, %zero)
        %query2 = prim::NumToTensor(%query1)
        %query3 = aten::size(%query0, %one)
        %query4 = aten::size(%query0, %two)
        %query5 = prim::NumToTensor(%query4)
        %query6 = aten::floor_divide(%query5, %headsize)
        %query7 = aten::Int(%query6)
        %querylist1 = prim::ListConstruct(%query1, %query3, %num_head, %query7)
        %query8 = aten::reshape(%query0, %querylist1)
        %query9 = aten::permute(%query8, %permutelist)
        %query10 = aten::mul(%query2, %num_head)
        %query11 = aten::Int(%query10)
        %querylist2 = prim::ListConstruct(%query11, %query3, %query7)
        %query = aten::reshape(%query9, %querylist2) )";

  std::string sd_mha_key = R"(
        %key1 = aten::size(%key0, %zero)
        %key2 = prim::NumToTensor(%key1)
        %key3 = aten::size(%key0, %one)
        %key4 = aten::size(%key0, %two)
        %key5 = prim::NumToTensor(%key4)
        %key6 = aten::floor_divide(%key5, %headsize)
        %key7 = aten::Int(%key6)
        %keylist1 = prim::ListConstruct(%key1, %key3, %num_head, %key7)
        %key8 = aten::reshape(%key0, %keylist1)
        %key9 = aten::permute(%key8, %permutelist)
        %key10 = aten::mul(%key2, %num_head)
        %key11 = aten::Int(%key10)
        %keylist2 = prim::ListConstruct(%key11, %key3, %key7)
        %key = aten::reshape(%key9, %keylist2) )";

  std::string sd_mha_value = R"(
        %value1 = aten::size(%value0, %zero)
        %value2 = prim::NumToTensor(%value1)
        %value3 = aten::size(%value0, %one)
        %value4 = aten::size(%value0, %two)
        %value5 = prim::NumToTensor(%value4)
        %value6 = aten::floor_divide(%value5, %headsize)
        %value7 = aten::Int(%value6)
        %valuelist1 = prim::ListConstruct(%value1, %value3, %num_head, %value7)
        %value8 = aten::reshape(%value0, %valuelist1)
        %value9 = aten::permute(%value8, %permutelist)
        %value10 = aten::mul(%value2, %num_head)
        %value11 = aten::Int(%value10)
        %valuelist2 = prim::ListConstruct(%value11, %value3, %value7)
        %value = aten::reshape(%value9, %valuelist2) )";

  std::string sd_mha_main_v1 = R"(
        %query_size1 = aten::size(%query, %zero)
        %query_size2 = aten::size(%query, %one)
        %key_size1 = aten::size(%key, %one)
        %emptylist = prim::ListConstruct(%query_size1, %query_size2, %key_size1)
        %baddbmm_input = aten::empty(%emptylist, %idx, %dtype, %device, %no, %dtype)
        %keytrans = aten::transpose(%key, %neg_one, %neg_two)
        %attention_scores = aten::baddbmm(%baddbmm_input, %query, %keytrans, %zero, %scale)
        %sm_out = ipex::softmax(%attention_scores, %neg_one, %dtype)
        %attention_probs = aten::to(%sm_out, %idx, %no, %no, %dtype)
        %bmm2 = aten::bmm(%attention_probs, %value)
        %size1 = aten::size(%bmm2, %zero)
        %size2 = prim::NumToTensor(%size1)
        %size3 = aten::size(%bmm2, %one)
        %size4 = aten::size(%bmm2, %two)
        %dim1 = prim::NumToTensor(%size4)
        %size5 = aten::floor_divide(%size2, %headsize)
        %size6 = aten::Int(%size5)
        %sizelist = prim::ListConstruct(%size6, %num_head, %size3, %size4)
        %out1 = aten::reshape(%bmm2, %sizelist)
        %out2 = aten::permute(%out1, %permutelist)
        %size7 = aten::mul(%dim1, %num_head)
        %size8 = aten::Int(%size7)
        %reshapelist = prim::ListConstruct(%size6, %size3, %size8)
        %output = aten::reshape(%out2, %reshapelist)
        return (%output) )";

  std::string sd_mha_main_v2 = R"(
        %viewlist = prim::ListConstruct(%batchsize, %neg_one, %num_head, %headsize)
        %query1 = aten::view(%query0, %viewlist)
        %query2 = aten::transpose(%query1, %one, %two)
        %key1 = aten::view(%key0, %viewlist)
        %key2 = aten::transpose(%key1, %one, %two)
        %value1 = aten::view(%value0, %viewlist)
        %value2 = aten::transpose(%value1, %one, %two)
        %hidden_states = aten::scaled_dot_product_attention(%query2, %key2, %value2, %dtype, %dropout, %no, %scale)
        %out0 = aten::transpose(%hidden_states, %one, %two)
        %reshapelist = prim::ListConstruct(%batchsize, %neg_one, %hiddensize)
        %out1 = aten::reshape(%out0, %reshapelist)
        %output = aten::to(%out1, %idx, %no, %no, %dtype)
        return (%output) )";

  std::string sd_fused_mha_main_v1 = R"(
        %output = ipex::sd_flash_mha(%qkv, %split_idx, %scale, %num_head)
        return (%output) )";

  std::string sd_fused_mha_main_v2 = R"(
        %output = ipex::sd_flash_mha(%query0, %key0, %value0, %scale, %num_head)
        return (%output) )";

  auto sd_mha_pattern_v1 = sd_mha_graph_v1 + sd_qkv_split + sd_mha_query +
      sd_mha_key + sd_mha_value + sd_mha_main_v1;
  auto sd_mha_pattern_v2 = sd_mha_graph_v2 + sd_mha_query + sd_mha_key +
      sd_mha_value + sd_mha_main_v1;
  auto sd_mha_pattern_v3 = sd_mha_graph_v3 + sd_qkv_split + sd_mha_main_v2;
  auto sd_mha_pattern_v4 = sd_mha_graph_v4 + sd_mha_main_v2;
  auto sd_fused_mha_pattern_v1 = sd_mha_graph_v1 + sd_fused_mha_main_v1;
  auto sd_fused_mha_pattern_v2 = sd_mha_graph_v2 + sd_fused_mha_main_v2;
  auto sd_fused_mha_pattern_v3 = sd_mha_graph_v3 + sd_fused_mha_main_v1;
  auto sd_fused_mha_pattern_v4 = sd_mha_graph_v4 + sd_fused_mha_main_v2;
  SubgraphRewriter sd_mha_fusion_v1, sd_mha_fusion_v2, sd_mha_fusion_v3,
      sd_mha_fusion_v4;
  sd_mha_fusion_v1.RegisterRewritePattern(
      sd_mha_pattern_v1, sd_fused_mha_pattern_v1);
  sd_mha_fusion_v1.runOnGraph(graph, sd_flash_mha_filter_v1);
  sd_mha_fusion_v2.RegisterRewritePattern(
      sd_mha_pattern_v2, sd_fused_mha_pattern_v2);
  sd_mha_fusion_v2.runOnGraph(graph, sd_flash_mha_filter_v2);
  // sd_mha_fusion_v3.RegisterRewritePattern(
  //     sd_mha_pattern_v3, sd_fused_mha_pattern_v3);
  // sd_mha_fusion_v3.runOnGraph(graph, sd_flash_mha_filter_v3);
  // sd_mha_fusion_v4.RegisterRewritePattern(
  //     sd_mha_pattern_v4, sd_fused_mha_pattern_v4);
  // sd_mha_fusion_v4.runOnGraph(graph, sd_flash_mha_filter_v4);

  auto bmm_pattern = R"(
    graph(%batch1, %batch2):
        %res = aten::matmul(%batch1, %batch2)
        return (%res))";
  std::string transfree_bmm_pattern = R"(
    graph(%batch1, %batch2):
        %res = ipex::matmul(%batch1, %batch2)
        return (%res))";

  SubgraphRewriter rewriter_bmm;
  rewriter_bmm.RegisterRewritePattern(bmm_pattern, transfree_bmm_pattern);
  rewriter_bmm.runOnGraph(graph, transfree_bmm_filter);

  std::string bmm_outtrans_graph_v1 = R"(
      graph(%bmm1: Tensor, %value_layer: Tensor, %permute: int[]): )";
  std::string bmm_outtrans_graph_v2 = R"(
      graph(%bmm1: Tensor, %value_layer: Tensor, %trans_a: int, %trans_b: int): )";
  std::string bmm2 = R"(
        %bmm2 = ipex::matmul(%bmm1, %value_layer) )";
  std::string bmm_outtrans_v1 = R"(
        %context_layer1  = aten::permute(%bmm2, %permute) )";
  std::string bmm_outtrans_v2 = R"(
        %context_layer1  = aten::transpose(%bmm2, %trans_a, %trans_b) )";
  std::string bmm_outtrans_output = R"(
        return (%context_layer1) )";

  std::string fused_bmm_outtrans = R"(
        %output = ipex::matmul_outtrans(%bmm1, %value_layer)
        return (%output) )";

  std::string bmm_outtrans_pattern_v1 =
      bmm_outtrans_graph_v1 + bmm2 + bmm_outtrans_v1 + bmm_outtrans_output;
  std::string bmm_outtrans_pattern_v2 =
      bmm_outtrans_graph_v2 + bmm2 + bmm_outtrans_v2 + bmm_outtrans_output;
  std::string fused_bmm_outtrans_pattern_v1 =
      bmm_outtrans_graph_v1 + fused_bmm_outtrans;
  std::string fused_bmm_outtrans_pattern_v2 =
      bmm_outtrans_graph_v2 + fused_bmm_outtrans;
  SubgraphRewriter bmm_outtrans_fusion_v1, bmm_outtrans_fusion_v2;
  bmm_outtrans_fusion_v1.RegisterRewritePattern(
      bmm_outtrans_pattern_v1, fused_bmm_outtrans_pattern_v1);
  bmm_outtrans_fusion_v1.runOnGraph(graph, bmm_outtrans_filter_v1);
  bmm_outtrans_fusion_v2.RegisterRewritePattern(
      bmm_outtrans_pattern_v2, fused_bmm_outtrans_pattern_v2);
  bmm_outtrans_fusion_v2.runOnGraph(graph, bmm_outtrans_filter_v2);
}
} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
