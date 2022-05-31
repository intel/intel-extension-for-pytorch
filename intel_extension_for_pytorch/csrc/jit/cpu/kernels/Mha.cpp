#include "Mha.h"
#include "Softmax.h"
#include "csrc/aten/cpu/AddSoftmax.h"
#include "csrc/aten/cpu/DivSoftmax.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/utils/ipex_op_profile.h"

namespace torch_ipex {
namespace cpu {

/**
 * We tried to fuse Div+Matmul+Add+Softmax as a signel operator. But
 * the oneDNN matmul performance with binary postop is poor, then we splited
 * the fusion into two parts - Div+Matmul and Add+Softmax. When the oneDNN
 * fixes the performance issue, we can directly leverage oneDNN's
 * implementation.
 **/
at::Tensor dil_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& rel_kv,
    const at::Scalar& alpha,
    const at::Scalar& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype) {
  IPEX_RECORD_FUNCTION("dil_mha_scores_calc", c10::ArrayRef<c10::IValue>({}));

  auto _dim_per_head = dim_per_head.to<float>();
  auto _alpha = alpha.to<float>();
  auto qk = at::Tensor();

  auto q_dim = q.dim();
  auto k_dim = k.dim();
  qk = at::matmul(q, k);

  // Only support last dimension
  bool is_last_dim = (softmax_dim == -1);
  // Only support the non-last-dimension broadcast
  bool not_last_dim_broadcast = (rel_kv.size(rel_kv.ndimension() - 1) != 1);
  // Only support >=2D
  bool not_one_dim = q_dim >= 2;
  // Only support contiguous tensor
  bool is_contiguous = rel_kv.is_contiguous() && qk.is_contiguous();
  if (is_last_dim && not_last_dim_broadcast && not_one_dim && is_contiguous &&
      dtype.isNone() && _alpha == 1.0f) {
    return DivAddSoftmax(qk, rel_kv, _dim_per_head);
  } else {
    qk = at::div(qk, dim_per_head);
    qk = at::add(qk, rel_kv, _alpha);
    return dil_softmax(qk, softmax_dim, dtype);
  }
}
/**
 * For BF16/FP32 path, We split the distil mha fusion into two parts - Matmul
 * and Div+Maskedfill+Softmax.
 * We do input checkings at graph rewrite time,
 * so we assume here:
 * Only support last dimension for softmax
 * Only support contiguous tensor for qk and mask
 * Only support qk.dim >=2D
 * Only support 64byte aligned
 * Only support when expand from the mid dims shape (bs :: seq_length)
 * Also checking the dtype as None
 **/
at::Tensor dil_distil_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const int64_t& transpose_dim_a,
    const int64_t& transpose_dim_b,
    const at::Scalar& fill,
    const at::Scalar& dim_per_head) {
  IPEX_RECORD_FUNCTION(
      "dil_distil_mha_scores_calc", c10::ArrayRef<c10::IValue>({}));
  auto _dim_per_head = dim_per_head.to<float>();
  auto _fill = fill.to<float>();
  auto qk = at::Tensor();
  auto _k = k.transpose(transpose_dim_a, transpose_dim_b);
  qk = at::matmul(q, _k);
  // convert the mask to float for creating vec mask for kernel computation
  auto _mask_qk = mask_qk.toType(at::kFloat);
  return DivMaskedfillSoftmax(
      qk, _mask_qk, mask_qk_reshp, _fill, _dim_per_head);
}

/**
 * For INT8 path, since matmul and div would be handled by LLGA fusion group,
 * We have to handle the rest fusion - Maskedfill+Softmax.
 * We also do the same input checkings at graph rewrite time like mentioned in
 * above overload function notes.
 **/
at::Tensor dil_maskedfill_softmax(
    at::Tensor& qk,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const at::Scalar& fill) {
  IPEX_RECORD_FUNCTION(
      "dil_maskedfill_softmax", c10::ArrayRef<c10::IValue>({}));
  float _dim_per_head = 1;
  auto _fill = fill.to<float>();
  // convert the mask to float for creating vec mask for kernel computation
  auto _mask_qk = mask_qk.toType(at::kFloat);
  return DivMaskedfillSoftmax(
      qk, _mask_qk, mask_qk_reshp, _fill, _dim_per_head);
}

} // namespace cpu
} // namespace torch_ipex
