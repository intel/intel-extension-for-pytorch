#include "Matmul.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/jit/cpu/kernels/Matmul.h"

namespace torch_ipex {
namespace cpu {

using at::IntArrayRef;
using at::Tensor;

//! function: is_add_broadcast_supported_by_onednn
/*!
 * This is a workaround checking since oneDNN is not well supported
 * matmul+binary_add fusion with all kinds of add input broadcast dims;
 * Depending the add input broadcast dims, oneDNN matmul+binary_add will go into
 * ref path in some cases; Here we add this function checking to map those
 * verified supported cases, and fallback those unsupported cases;
 *
 * The verified supported cases use following oneDNN non_broadcast_mask:
 * 2D: oneDNN non_broadcast_mask = {0, 2, 3}
 * 3D: oneDNN non_broadcast_mask = {0, 2, 4, 5, 7}
 * 4D: oneDNN non_broadcast_mask = {0, 2, 8, 9, 13, 15}
 *
 * For example:
 * For 4D tensors, left has shape [8, 2, 4, 6] and right has shape [8, 2, 6, 4],
 * so matmul shape is [8, 2, 4, 4], and post_add_tensor has shape [8, 1, 1, 4].
 * Therefore, the according non_broadcast_mask is 9, which is supported.
 *
 * \param left: the left operand of matmul
 * \param right: the right operand of matmul
 * \param post_add_tensor: the post add input tensor
 * \return: whether the post add input is supported for broadcast by oneDNN for
 * matmul+binary_add fusion
 */
bool is_add_broadcast_supported_by_onednn(
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& post_add_tensor) {
  // we only support add.dim == left.dim == right.dim == output.dim
  // for that we can not enumerate all the cases when output.dim is reduced and
  // testing the ability of oneDNN kernel for this workaround
  if (post_add_tensor.dim() != left.dim() ||
      post_add_tensor.dim() != right.dim()) {
    return false;
  }
  auto non_broadcast_mask = 0;
  for (int i = 0; i < left.dim(); i++) {
    if (post_add_tensor.size(i) != 1) {
      if (i == left.dim() - 1) {
        non_broadcast_mask +=
            post_add_tensor.size(i) == right.size(i) ? 1 << i : 0;
      } else {
        non_broadcast_mask +=
            post_add_tensor.size(i) == left.size(i) ? 1 << i : 0;
      }
    }
  }
  if (left.dim() == 4) {
    if (non_broadcast_mask == 0 || non_broadcast_mask == 2 ||
        non_broadcast_mask == 8 || non_broadcast_mask == 9 ||
        non_broadcast_mask == 13 || non_broadcast_mask == 15) {
      return true;
    }
  } else if (left.dim() == 3) {
    if (non_broadcast_mask == 0 || non_broadcast_mask == 2 ||
        non_broadcast_mask == 4 || non_broadcast_mask == 5 ||
        non_broadcast_mask == 7) {
      return true;
    }
  } else if (left.dim() == 2) {
    if (non_broadcast_mask == 0 || non_broadcast_mask == 2 ||
        non_broadcast_mask == 3) {
      return true;
    }
  }

  return false;
}
//! function: sumproduct_pair
/*!
 *
 * Use oneDNN matmul to do the matrix multiplication and post-ops to
 * fuse binary post-ops. ToDo: Enable other binary ops instead of only add
 * 1) The input1 will be [lro, lo, sum_dims, ro](ro is all 1 for input1), while
 * the input2 will be [lro, sum_dims, ro, lo] (lo is all 1 for input2 and the
 * intermediate output of matmul will be [lro, lo, ro].
 *
 * 2) According to the oneDNN matmul data representation of ND data. we should
 * collapse the ro dims(output channels) in the input2 and s_um_dims_ in both
 * input1 and input2, and need to adapt the ro dims in input1 and lo dims in the
 * input2. such as remove the rightest dims if they are both 1 for input1(ro
 * dims in input1) and input2(lo dims in input2).
 *
 * 3) To fused the add with matmul post-ops. we need to adapt the add args to be
 * able to broadcast with the output of oneDNN matmul(either to have same shape
 * as output or to be 1 in the corresponding dim). a). unsqueeze and permute the
 * add arg to have the same number of dims as the output [lro, lo, ro] b).expand
 * the add arg to have the same ro dims as input2. c). collapse the 'ro' dims of
 * add arg.
 *
 * 4) Adapt the output of matmul to be the shape of the eninsum output.
 * \param left_: the left operand of matmul
 * \param right_: the right operand of matmul
 * \param sum_dim_: the reduction dim info
 * \param keep_dim: keep the target dim of output
 * \param udi_left unsqueezed info for every dim
 * \param udi_right unsqueezed info for every dim
 * \param add_arg: the other input of binary post-op
 * \param alpha: the multiplier for other.
 * \return: the ouput of einsum+binary
 */
static Tensor sumproduct_pair(
    const Tensor& left_,
    const Tensor& right_,
    IntArrayRef sum_dims_,
    bool keepdim,
    const std::vector<bool> udi_left,
    const std::vector<bool> udi_right,
    const Tensor& add_arg,
    const c10::Scalar& alpha) {
  // assumes that tensors have been pre-unsqueezed (so that all dimensions match
  // - after broadcasting) but makes no other assumptions on the order of
  // dimensions
  TORCH_CHECK(left_.dim() == right_.dim(), "number of dimensions must match");
  int64_t dim = left_.dim();
  auto sum_dims = at::dim_list_to_bitset(sum_dims_, dim);
  // dimensions that will be part of the output (i.e. not summed over) in three
  // vectors dims in lro appear in left, right and output, similarly lo: left
  // and output, ro: right and output also the sizes are kept track of for
  // reshaping
  std::vector<int64_t> lro, lo, ro;
  int64_t lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  Tensor left = left_;
  Tensor right = right_;
  for (const auto i : c10::irange(dim)) {
    auto sl = left.size(i) > 1 || (left.size(i) == 1 && udi_left[i] == 0);
    auto sr = right.size(i) > 1 || (right.size(i) == 1 && udi_right[i] == 0);
    if (sum_dims[i]) { // first dimensions that will be summed over after
                       // multiplication
      if (sl && sr) { // dimensions nontrivially in both left and right must be
                      // of the same size
        TORCH_CHECK(
            left.size(i) == right.size(i),
            "non-broadcast dimensions must match");
        sum_size *= left.size(i);
      } else if (sl) { // if it is only in one of left and right, we can sum
                       // right away
        left = left.sum(i, true);
      } else if (sr) {
        right = right.sum(i, true);
      }
    } else if (sl && sr) { // now deal with dimensions  dimensions that will be
                           // in the output
      // dimensions nontrivially in both left and right must be of the same size
      TORCH_CHECK(
          left.size(i) == right.size(i), "non-broadcast dimensions must match");
      lro.push_back(i);
      lro_size *= left.size(i);
    } else if (sl) { // keep track of dimensions appearing only once
      lo.push_back(i);
      lo_size *= left.size(i);
    } else {
      ro.push_back(i);
      ro_size *= right.size(i);
    }
  }
  // we now work with the following permutations / shapes.
  // the pipeline is permute inputs -> reshape inputs -> batch matrix mul ->
  // reshape(view) output -> permute output output: "lro, lo, 1-for-summed-dims,
  // ro" with orgiginal shape dimensions left:   "lro, lo, summed" permuted with
  // lpermutation and the three flattened right:  "lro, summed, ro" permuted
  // with rpermutation and the three flattened then the permuted output is a
  // view of bmm(left, right) finally, opermutation reverts the permutation to
  // the original order of dimensions
  std::vector<int64_t> out_size;
  // NOLINTNEXTLINE(performance-inefficient-vector-operation)
  for (auto& d : lro)
    out_size.push_back(left.size(d));
  for (auto& d : lo)
    out_size.push_back(left.size(d));
  for (auto& d : sum_dims_) {
    out_size.push_back(1);
    (void)(d);
  }; // avoid warining about not using d
  for (auto& d : ro)
    out_size.push_back(right.size(d));
  std::vector<int64_t> lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  std::vector<int64_t> rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  std::vector<int64_t> argpermutation(lro);
  argpermutation.insert(argpermutation.end(), lo.begin(), lo.end());
  argpermutation.insert(argpermutation.end(), ro.begin(), ro.end());

  std::vector<int64_t> opermutation(
      lro.size() + lo.size() + sum_dims_.size() + ro.size(), -1);
  {
    int64_t i = 0;

    for (auto it = lro.cbegin(); it != lro.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = lo.cbegin(); it != lo.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = sum_dims_.cbegin(); it != sum_dims_.cend(); i++, it++) {
      opermutation[*it] = i;
    }
    for (auto it = ro.cbegin(); it != ro.cend(); i++, it++) {
      opermutation[*it] = i;
    }
  }

  // unsqueeze the dim of add_arg corresponding to the dim of einsum result
  // eg: the shape of result is [64,128,1024] while the shape of add_arg is
  // [1024] we should unsqueeze add_arg to [1,1,1024] to use oneDNN binary ops.
  Tensor arg = add_arg;
  auto out_dims = dim - sum_dims_.size();
  auto add_arg_dim = add_arg.dim();

  for (int i = out_dims - 1; i >= 0; i--) {
    if (add_arg_dim-- <= 0)
      arg = arg.unsqueeze(0);
  }
  arg = arg.permute(argpermutation);

  // 1)expand the add_arg to the same shape in ro dims
  // 2)collapse the ro dims to use the oneDNN matmul, pls see the
  // https://oneapi-src.github.io/oneDNN/dev_guide_matmul.html#data-representation
  //  Dims  Source   Weights  Destination    Bias
  //   ND   SxMxK     WxKxN    DxMxN       None or B
  //  where S and W means the batch dims for input1, it can be multi-dims and
  //  they should have same dim order. B should also have the same dim order as
  //  the output for every dim, they should be same or to be 1 to make sure they
  //  can be broadcasted.
  if (ro.size() > 1) {
    auto arg_size = arg.sizes();
    std::vector<int64_t> expanded_size(out_dims, 1);
    for (int i = 0; i < out_dims - ro.size(); i++) {
      expanded_size[i] = arg_size[i];
    }
    // get the expand size according to the ro informantion
    for (int i = 0; i < ro.size(); i++) {
      expanded_size[out_dims - ro.size() + i] = right.size(ro[i]);
    }
    arg = arg.expand(expanded_size);
    arg_size = arg.sizes();
    // we need to firstly collapse ro dims for the arg
    std::vector<int64_t> collapsed_size(out_dims - ro.size() + 1, 1);
    for (int i = 0; i < out_dims - ro.size(); i++) {
      collapsed_size[i] = arg_size[i];
    }
    for (int i = 1; i <= ro.size(); i++) {
      collapsed_size[out_dims - ro.size()] *= arg_size[out_dims - i];
    }

    arg = arg.reshape(collapsed_size);
  }
  std::vector<int64_t> left_shape, right_shape;
  for (auto i : lpermutation) {
    left_shape.push_back(left.size(i));
  }
  for (auto i : rpermutation) {
    right_shape.push_back(right.size(i));
  }
  auto lo_count = lo.size();
  auto ro_count = ro.size();
  if (ro_count > 1) { //
    auto i = 1;
    for (; i <= ro_count - 1; i++) {
      right_shape[dim - lo_count - i] = 1;
    }
    right_shape[dim - lo_count - i] = ro_size;
  }
  auto lro_count = lro.size();
  auto min_size = std::min(lo_count, ro_count);
  auto max_size = std::max(lo_count, ro_count);
  for (int i = dim - 1; i >= dim - max_size; i--) {
    if (left_shape[i] == 1 && left_shape[i] == right_shape[i] &&
        min_size >= 0) {
      right_shape.pop_back();
      left_shape.pop_back();
      min_size--;
    } else if (min_size == 0 && lo_count > ro_count) {
      auto diff_dim = lo_count - ro_count;
      for (auto j = 0; j < diff_dim; j++) {
        right_shape.pop_back();
        right_shape.insert(right_shape.begin() + lro_count, 1);
      }
    } else {
      auto diff_dim = ro_count - lo_count;
      while (diff_dim-- && i >= dim - max_size) {
        left_shape.insert(left_shape.begin() + lro_count, 1);
        left_shape.pop_back();
        i--;
      }
    }
  }
  if (ro_count > 1) {
    while (right_shape[right_shape.size() - 1] == 1) {
      right_shape.pop_back();
      right_shape.insert(right_shape.begin() + lro_count, 1);
    }
  }
  if (sum_dims_.size() > 1) { // need to collapse the sum dims
    auto matmul_dims = left_shape.size();
    left_shape[matmul_dims - sum_dims_.size()] = sum_size;
    right_shape[matmul_dims - sum_dims_.size() - 1] = sum_size;
    for (auto i = 1; i < sum_dims_.size(); i++) {
      left_shape.erase(
          left_shape.begin() + (matmul_dims - sum_dims_.size()) + 1);
      right_shape.erase(right_shape.begin() + (matmul_dims - sum_dims_.size()));
    }
  }
  // now we can execute the operations above
  left = left.permute(lpermutation).reshape(left_shape);
  right = right.permute(rpermutation).reshape(right_shape);

  // now we do the computation
  Tensor result;
  bool is_fallback_post_add = false;
  if (is_add_broadcast_supported_by_onednn(left, right, arg)) {
    auto _input = arg.is_contiguous() ? arg : arg.contiguous();
    ideep::tensor onednn_input = itensor_view_from_dense(_input);
    auto op_attr = ideep::attr_t::fuse_binary(
        dnnl::algorithm::binary_add, onednn_input.get_desc());
    result = bmm_impl(left, right, at::Tensor(), op_attr, {onednn_input}, 1.0f);
  } else {
    result = at::matmul(left, right);
    is_fallback_post_add = true;
  }

  result = result.view(out_size).permute(opermutation);

  // finally squeeze summed dimensions if desired
  if (!keepdim) {
    auto sizes = result.sizes().vec();
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    for (int i = dim - 1; i >= 0; i--) {
      if (sum_dims[i]) {
        sizes.erase(sizes.begin() + i);
      }
    }
    result = result.view(sizes);
  }

  // if fallback, add op should be done after einsum has finalize the result,
  // like the result may have another view if "keepdim" is false
  if (is_fallback_post_add) {
    auto f_alpha = alpha.to<float>();
    result = result + f_alpha * add_arg;
  }
  return result;
}

namespace {

bool einsum_check_label(unsigned char label) {
  return std::isalpha(label);
}

uint8_t einsum_label_to_index(unsigned char label) {
  constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
  return std::isupper(label) ? label - 'A' : NUM_OF_LETTERS + (label - 'a');
}

unsigned char einsum_index_to_label(uint8_t index) {
  constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
  return index < NUM_OF_LETTERS ? index + 'A' : index - NUM_OF_LETTERS + 'a';
}

} // namespace

//! function: einsum_prepare
/*!
 *This function do the following preparation:
 *1) parse the einsum equation to get inputs/output info
 *2) unsqueeze and permute the inputs/output to have same dims. The dim order
 * of all inputs and output is same.
 *\param equation:  The subscripts for the Einstein summation.
 *more detials about equation can found:
 *https://pytorch.org/docs/stable/generated/torch.einsum.html
 *\param operands: The tensors to compute the Einstein summation of.
 *\return tuple<has_zero_size_dim, out_size, dim_last_op, sum_dims,
 *permuted_operands, unsqueezed_dim_info>
 */
std::tuple<
    bool,
    int64_t,
    std::vector<std::size_t>,
    std::vector<int64_t>,
    std::vector<Tensor>,
    std::vector<std::vector<bool>>>
einsum_prepare(
    c10::string_view equation,
    const c10::List<at::Tensor>& operands) {
  // Code used to identify ELLIPSIS ("...")
  constexpr uint8_t ELLIPSIS = 52;

  // Find arrow (->) to split equation into lhs and rhs
  const auto arrow_pos = equation.find("->");
  const auto lhs = equation.substr(0, arrow_pos);

  const auto num_ops = operands.size();

  // Convert labels for input operands into an index in [0, 52) and store
  // them in op_labels for each operand along with ELLIPSIS if present.
  std::vector<std::vector<uint8_t>> op_labels(num_ops);
  bool found_ell = false;
  std::size_t curr_op = 0;
  for (auto i = decltype(lhs.length()){0}; i < lhs.length(); ++i) {
    const unsigned char label = lhs[i];
    switch (label) {
      case ' ':
        // Ignore spaces
        break;

      case '.':
        TORCH_CHECK(
            // Only one ellipsis per operand can be given
            !found_ell,
            "einsum(): found \'.\' for operand ",
            curr_op,
            " for which an ellipsis was already found");
        TORCH_CHECK(
            // Ensure it's a valid ellipsis
            i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.',
            "einsum(): found \'.\' for operand ",
            curr_op,
            " that is not part of any ellipsis");
        op_labels[curr_op].push_back(ELLIPSIS);
        found_ell = true;
        break;

      case ',':
        // Move onto next operand
        ++curr_op;
        TORCH_CHECK(
            curr_op < num_ops,
            "einsum(): fewer operands were provided than specified in the equation");
        found_ell = false;
        break;

      default:
        // Parse label
        TORCH_CHECK(
            einsum_check_label(label),
            "einsum(): invalid subscript given at index ",
            i,
            " in the equation string, subscripts must be in [a-zA-Z]");
        op_labels[curr_op].push_back(einsum_label_to_index(label));
    }
  }

  TORCH_CHECK(
      curr_op == num_ops - 1,
      "einsum(): more operands were provided than specified in the equation");

  // Labels must be within [a-zA-Z].
  constexpr uint8_t TOTAL_LABELS = 52;
  std::vector<int64_t> label_count(TOTAL_LABELS, 0);

  // The maximum number of dimensions covered by any ellipsis, needed when
  // unsqueezing missing dimensions from operands to permute and broadcast
  int64_t ell_num_dim = 0;

  // Compute label frequency and number of dimensions covered by ellipsis
  // We do this after parsing labels to make it more readable and simpler
  // to compute the number of dimensions covered by ellipsis.
  for (const auto i : c10::irange(num_ops)) {
    const auto operand = operands[i];
    const auto labels = op_labels[i];
    const auto ndims = operand.dim();
    int64_t nlabels = static_cast<int64_t>(labels.size());
    bool has_ellipsis = false;

    for (const auto& label : labels) {
      if (label == ELLIPSIS) {
        --nlabels;
        has_ellipsis = true;
        ell_num_dim = std::max(ell_num_dim, ndims - nlabels);
      } else {
        ++label_count[label];
      }
    }

    TORCH_CHECK(
        has_ellipsis ? nlabels <= ndims : nlabels == ndims,
        "einsum(): the number of subscripts in the equation (",
        nlabels,
        has_ellipsis ? ") is more than the number of dimensions ("
                     : ") does not match the number of dimensions (",
        ndims,
        ") for operand ",
        i,
        has_ellipsis ? "" : " and no ellipsis was given");
  }

  // We want to align the dimensions of every input tensor to have
  // shape out_dims + sum_dims. For this, we create a mapping of label
  // to index into the permuted shape.
  std::vector<int64_t> label_perm_index(TOTAL_LABELS, -1);

  // Current index in the permuted shape
  int64_t perm_index = 0;

  // Start index of ellipsis dimensions in the permuted shape
  int64_t ell_index = 0;
  found_ell = false;

  if (arrow_pos == std::string::npos) {
    // Implicit output is ellipsis (...) + labels seen only once
    perm_index = ell_num_dim;
    found_ell = true;
    for (const auto label : c10::irange(TOTAL_LABELS)) {
      if (label_count[label] == 1) {
        label_perm_index[label] = perm_index++;
      }
    }
  } else {
    // Parse explicit output
    const auto rhs = equation.substr(arrow_pos + 2);
    for (auto i = decltype(rhs.length()){0}; i < rhs.length(); ++i) {
      const unsigned char label = rhs[i];
      switch (label) {
        case ' ':
          // Ignore spaces
          break;

        case '.':
          TORCH_CHECK(
              // There can only be one ellipsis in the output
              !found_ell,
              "einsum(): found \'.\' for output but an ellipsis (...) was already found");
          TORCH_CHECK(
              // Ensure ellipsis is correct
              i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.',
              "einsum(): found \'.\' for output that is not part of any ellipsis (...)");
          ell_index = perm_index;
          perm_index += ell_num_dim;
          found_ell = true;
          break;

        default:
          TORCH_CHECK(
              einsum_check_label(label),
              "einsum(): invalid subscript given at index ",
              lhs.size() + 2 + i,
              " in the equation string, subscripts must be in [a-zA-Z]");
          const auto index = einsum_label_to_index(label);
          TORCH_CHECK(
              // Ensure label appeared at least once for some input operand and
              // at most once for the output
              label_count[index] > 0 && label_perm_index[index] == -1,
              "einsum(): output subscript ",
              label,
              label_perm_index[index] > -1
                  ? " appears more than once in the output"
                  : " does not appear in the equation for any input operand");
          label_perm_index[index] = perm_index++;
      }
    }
  }

  // Save output size before adding contraction dims (dims to sum out)
  const int64_t out_size = perm_index;

  // If ellipsis is not part of the output, add to contraction dimensions
  if (!found_ell) {
    ell_index = perm_index;
    perm_index += ell_num_dim;
  }

  // Add contraction labels (labels not present in output)
  for (const auto label : c10::irange(TOTAL_LABELS)) {
    if (label_count[label] > 0 && label_perm_index[label] == -1) {
      label_perm_index[label] = perm_index++;
    }
  }

  // Here we unsqueeze missing dimensions to make all operands have the same
  // number of dimensions. We take diagonals for repeated labels within the
  // same operand. Finally we permute the operands to align dimensions as
  // per the perm_out_index we computed above.
  std::vector<Tensor> permuted_operands;

  // for the dim where size=1, should know whthere it is unsqueezed.
  // eg: (12,1,4,16) & (12,4,4,16)
  auto unsqueezed_dim_info =
      std::vector<std::vector<bool>>(2, std::vector<bool>(perm_index, 0));
  for (const auto i : c10::irange(num_ops)) {
    std::vector<int64_t> perm_shape(perm_index, -1);
    std::vector<int64_t> label_dim(TOTAL_LABELS, -1);
    Tensor operand = operands[i];
    const auto labels = op_labels[i];
    const auto original_sizes = operand.sizes();

    int64_t j = 0;
    for (const auto& label : labels) {
      if (label == ELLIPSIS) {
        // Add missing dimensions covered by the ellipsis
        const auto num_missing_dim =
            ell_num_dim - (original_sizes.size() - labels.size() + 1);
        for (const auto k : c10::irange(num_missing_dim)) {
          (void)k; // Suppress unused warning
          operand = operand.unsqueeze(j);
        }
        for (const auto k : c10::irange(ell_num_dim)) {
          perm_shape[ell_index + k] = j++;
        }
      } else if (label_dim[label] != -1) {
        // Repeated label, take diagonal
        const auto dim = label_dim[label];
        TORCH_CHECK(
            operand.size(j) == operand.size(dim),
            "einsum(): subscript ",
            einsum_index_to_label(label),
            " is repeated for operand ",
            i,
            " but the sizes don't match, ",
            operand.size(j),
            " != ",
            operand.size(dim));
        operand = operand.diagonal(0, dim, j).movedim(-1, dim);
      } else {
        // Lookup output index for label
        label_dim[label] = j;
        perm_shape[label_perm_index[label]] = j++;
      }
    }

    for (auto ind = 0; ind < perm_index; ind++) {
      if (perm_shape[ind] == -1) {
        unsqueezed_dim_info[i][ind] = 1;
      }
    }
    // Add dimensions for missing labels
    for (int64_t& index : perm_shape) {
      if (index == -1) {
        operand = operand.unsqueeze(-1);
        index = j++;
      }
    }
    permuted_operands.push_back(operand.permute(perm_shape));
  }

  // Check if operands broadcast and keep track of last operand with
  // dimension size != 1 for optimizing reductions
  std::vector<std::size_t> dim_last_op(perm_index, 0);
  bool has_zero_size_dim = false;
  for (const auto dim : c10::irange(perm_index)) {
    auto broadcast_size = permuted_operands[0].size(dim);
    for (const auto i : c10::irange(1, num_ops)) {
      const auto dim_size = permuted_operands[i].size(dim);
      if (broadcast_size != dim_size && broadcast_size != 1 && dim_size != 1) {
        std::ostringstream msg;
        msg << "einsum(): operands do not broadcast with remapped shapes [original->remapped]:";
        for (const auto j : c10::irange(num_ops)) {
          msg << " " << operands[j].sizes() << "->"
              << permuted_operands[j].sizes();
        }
        TORCH_CHECK(false, msg.str());
      }
      if (dim_size != 1) {
        broadcast_size = dim_size;
        dim_last_op[dim] = i;
      }
    }
    has_zero_size_dim |= broadcast_size == 0;
  }

  // Compute result
  // Sum out or squeeze dimensions that are size 1 for all later operands
  int64_t dim = out_size;
  for (int64_t i = dim; i < perm_index; ++i, ++dim) {
    if (dim_last_op[i] == 0) {
      if (permuted_operands[0].size(dim) == 1) {
        permuted_operands[0] = permuted_operands[0].squeeze(dim--);
      } else {
        permuted_operands[0] = permuted_operands[0].sum(dim--);
      }
    }
  }

  // we only process two operands, so the operands index is from [0, 1]
  std::vector<int64_t> sum_dims;
  // Sum out or squeeze dimensions that are size 1 for all later operands
  dim = out_size;
  for (int64_t j = dim; j < perm_index; ++j, ++dim) {
    if (dim_last_op[j] < 1) {
      permuted_operands[1] = permuted_operands[1].squeeze(dim);
      --dim;
    } else if (dim_last_op[j] == 1) {
      if (permuted_operands[0].size(dim) == 1) {
        permuted_operands[1] = permuted_operands[1].sum(dim);
        permuted_operands[0] = permuted_operands[0].squeeze(dim);
        --dim;
      } else {
        sum_dims.push_back(dim);
      }
    }
  }

  return std::make_tuple(
      has_zero_size_dim,
      out_size,
      dim_last_op,
      sum_dims,
      permuted_operands,
      unsqueezed_dim_info);
}

//! function: einsum_binary
/*!
 * This function use oneDNN binary post-ops to do the einsum+binary fusion.
 *\param equation:  The subscripts for the Einstein summation.
 *more detials about equation can found:
 *https://pytorch.org/docs/stable/generated/torch.einsum.html
 *\param operands: The tensors to compute the Einstein summation of.
 *\param add_arg: the other input of binary ops.
 *\param alpha: the multiplier for other.
 */
at::Tensor einsum_binary(
    c10::string_view equation,
    const c10::List<at::Tensor>& operands,
    const at::Tensor& add_arg,
    const c10::Scalar& alpha) {
  RECORD_FUNCTION("dil_einsum_binary", c10::ArrayRef<c10::IValue>({}));
  auto prepare_res = einsum_prepare(equation, operands);
  bool has_zero_size_dim = std::get<0>(prepare_res);
  auto out_size = std::get<1>(prepare_res);
  auto dim_last_op = std::get<2>(prepare_res);
  auto sum_dims = std::get<3>(prepare_res);
  auto permuted_operands = std::get<4>(prepare_res);
  auto unsqueezed_dim_info = std::get<5>(prepare_res);
  Tensor result = permuted_operands[0];
  Tensor operand = permuted_operands[1];
  std::vector<bool> udi_result = unsqueezed_dim_info[0];
  std::vector<bool> udi_operand = unsqueezed_dim_info[1];
  auto f_alpha = alpha.to<float>();

  // Fast path for when an operand has zero sized dim
  if (has_zero_size_dim) {
    std::vector<int64_t> out_shape(out_size);
    for (const auto i : c10::irange(out_size)) {
      out_shape[i] = permuted_operands[dim_last_op[i]].size(i);
    }
    return at::zeros(out_shape, result.options()) + f_alpha * add_arg;
  }

  // Multiply tensors and sum out dimensions in sum_dims
  if (sum_dims.empty()) {
    result = result.mul(operand);
    result = result + f_alpha * add_arg;
  } else if (sum_dims.size() == result.sizes().size()) {
    result = result.flatten().dot(operand.flatten());
    result = result + f_alpha * add_arg;
  } else {
    result = sumproduct_pair(
        result,
        operand,
        sum_dims,
        false,
        udi_result,
        udi_operand,
        add_arg,
        alpha);
  }

  return result;
}

} // namespace cpu
} // namespace torch_ipex
