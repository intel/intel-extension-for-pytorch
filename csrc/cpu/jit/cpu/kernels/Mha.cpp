#include "Mha.h"
#include "Matmul.h"
#include "Softmax.h"
#include "aten/AddSoftmax.h"
#include "aten/DivSoftmax.h"
#include "aten/MultiHeadAttention.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

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
  RECORD_FUNCTION("dil_mha_scores_calc", c10::ArrayRef<c10::IValue>({}));
  auto _dim_per_head = dim_per_head.to<float>();
  auto _alpha = alpha.to<float>();
  auto qk = at::Tensor();

  auto q_dim = q.dim();
  auto k_dim = k.dim();
  qk = bmm_impl(q, k, qk, ideep::attr_t(), {}, 1.f);

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
 * and Div+2DMaskedfill+Softmax.
 * We do input checkings at graph rewrite time,
 * so we assume here:
 * Only support last dimension for softmax
 * Only support contiguous tensor for qk and mask
 * Only support qk.dim >=2D
 * Only support when expand from the mid dims shape (bs :: seq_length)
 * Also checking the dtype as None
 **/
at::Tensor dil_distil_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const at::Scalar& fill,
    const at::Scalar& dim_per_head) {
  RECORD_FUNCTION("dil_distil_mha_scores_calc", c10::ArrayRef<c10::IValue>({}));
  auto _dim_per_head = dim_per_head.to<float>();
  auto _fill = fill.to<float>();
  auto qk = at::Tensor();
  qk = bmm_impl(q, k, qk, ideep::attr_t(), {}, 1.f);
  //  convert the mask to float for creating vec mask for kernel computation
  auto _mask_qk = mask_qk.toType(at::kFloat);
  return DivMaskedfillSoftmax(
      qk, _mask_qk, mask_qk_reshp, _fill, _dim_per_head);
}

/**
 * For BF16/FP32 path, We split the vit (ats_vit) mha fusion into two parts -
 * Matmul and Div+4DMaskedfill+Softmax.
 * We do input checkings at graph rewrite time, so we assume here:
 * Only support last dimension for softmax
 * Only support contiguous tensor for qk and mask
 * Only support qk.dim >=2D
 * Only support mask has the same dim as qk (broadcastable)
 * Also checking the dtype as None
 **/
at::Tensor dil_vit_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& mask_qk_reshp,
    const at::Scalar& fill,
    const at::Scalar& dim_per_head) {
  RECORD_FUNCTION("dil_vit_mha_scores_calc", c10::ArrayRef<c10::IValue>({}));
  auto _dim_per_head = 1 / dim_per_head.to<float>();
  auto _fill = fill.to<float>();
  auto qk = at::Tensor();
  qk = bmm_impl(q, k, qk, ideep::attr_t(), {}, 1.f);
  //  convert the mask to float for creating vec mask for kernel computation
  auto _mask_qk = mask_qk_reshp.toType(at::kFloat);
  return DivMaskedfillSoftmax(qk, _mask_qk, {}, _fill, _dim_per_head);
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
  RECORD_FUNCTION("dil_maskedfill_softmax", c10::ArrayRef<c10::IValue>({}));
  float _dim_per_head = 1;
  auto _fill = fill.to<float>();
  // convert the mask to float for creating vec mask for kernel computation
  auto _mask_qk = mask_qk.toType(at::kFloat);
  return DivMaskedfillSoftmax(
      qk, _mask_qk, mask_qk_reshp, _fill, _dim_per_head);
}

at::Tensor dil_transfree_vit_mha(
    const at::Tensor& qkv,
    const at::Tensor& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& num_head,
    const int64_t& headSize) {
  auto scale = dim_per_head.data_ptr<float>()[0];
  return dil_transfree_vit_mha(
      qkv, scale, softmax_dim, dtype, num_head, headSize);
}

at::Tensor dil_transfree_vit_mha(
    const at::Tensor& qkv,
    const double& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& num_head,
    const int64_t& headSize) {
  RECORD_FUNCTION("dil_transfree_vit_mha", c10::ArrayRef<c10::IValue>({}));

  int64_t batchSize = qkv.dim() > 2 ? qkv.size(0) : 1;
  int64_t sequenceSize = qkv.dim() > 2 ? qkv.size(1) : qkv.size(0);
  int64_t hiddenSize = num_head * headSize;
  at::Tensor qk =
      at::empty({batchSize, num_head, sequenceSize, sequenceSize}, qkv.dtype());

  auto qkv_mat = dil_mat_split<at::BFloat16>(
      qkv, at::IntArrayRef({hiddenSize, hiddenSize, hiddenSize}));
  auto query = qkv_mat[0];
  auto key = qkv_mat[1];
  auto value = qkv_mat[2];
  query.resize_({batchSize, sequenceSize, num_head, headSize}).transpose_(1, 2);
  key.resize_({batchSize, sequenceSize, num_head, headSize})
      .transpose_(1, 2)
      .transpose_(2, 3);
  value.resize_({batchSize, sequenceSize, num_head, headSize}).transpose_(1, 2);

  bmm_impl(query, key, qk, ideep::attr_t(), {}, dim_per_head);
  qk = dil_softmax_(qk, softmax_dim, dtype);

  auto output = dil_mha_matmul_trans(qk, value);

  return output;
}

/**
 * This BMM OP is designed for the second Batched-Matmul of MHA as
 * the output is fused with transpose OP.
 * All the tensors should be 4-dim and the transpose indices of the
 * output tensor be (1, 2).
 * If the tensors do not meet the above conditions, the performance
 * of bmm_impl may drop when it uses the DNNL Matmul primitive
 * as its backend.
 */
at::Tensor dil_mha_matmul_trans(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2) {
  RECORD_FUNCTION("dil_mha_bmm", c10::ArrayRef<c10::IValue>({}));

  std::vector<int64_t> output_size = {
      tensor1.size(0), tensor1.size(2), tensor1.size(1), tensor2.size(-1)};

  auto out = at::empty(output_size, tensor1.options()).transpose(1, 2);
  out = bmm_impl(tensor1, tensor2, out, ideep::attr_t(), {}, 1.f)
            .transpose_(1, 2);

  return out;
}

/**
 *  This kernel implements Flast attention
 * (https://hazyresearch.stanford.edu/blog/2023-01-12-flashattention-long-sequences)
 * on Bert models for BF16 dtype
 */
at::Tensor dil_bert_flash_mha(
    const at::Tensor& qkv,
    const at::Tensor& rel_kv,
    const at::Scalar& alpha,
    const at::Scalar& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& num_head,
    const int64_t& headSize) {
  RECORD_FUNCTION("dil_bert_flash_mha", c10::ArrayRef<c10::IValue>({}));
  auto _dim_per_head = dim_per_head.to<float>();
  return bert_flash_mha(qkv, rel_kv, num_head, headSize, _dim_per_head);
}

/**
 *  This kernel implements Flast attention on stable-diffusion models (from
 * Diffusers 0.12.1 and 0.13) for BF16 dtype, where qkv is from one
 * aten::linear; Note that in 0.13, aten::scaled_dot_product_attention uses the
 * scale of sqrt(headSize) if no scale is provided for query, where we are
 * following
 */
at::Tensor dil_sd_flash_mha(
    const at::Tensor& qkv,
    const at::IntArrayRef& split_list,
    const at::IValue& scale,
    const int64_t& num_head) {
  RECORD_FUNCTION("dil_sd_flash_mha_v1", c10::ArrayRef<c10::IValue>({}));
  int64_t headSize = qkv.size(-1) / split_list.size() / num_head;
  if (!scale.isNone()) {
    return sd_flash_mha(qkv, num_head, headSize, scale.toDouble());
  } else {
    auto scale_ = 1.f / sqrt(headSize);
    return sd_flash_mha(qkv, num_head, headSize, scale_);
  }
}

/**
 *  This kernel implements Flast attention on stable-diffusion models (from
 * Diffusers 0.12.1 and 0.13) for BF16 dtype, where qkv is splited; Note that
 * in 0.13, aten::scaled_dot_product_attention uses the scale of sqrt(headSize)
 * if no scale is provided for query, where we are following
 */
at::Tensor dil_sd_flash_mha(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::IValue& scale,
    const int64_t& num_head) {
  RECORD_FUNCTION("dil_sd_flash_mha_v2", c10::ArrayRef<c10::IValue>({}));
  int64_t headSize = query.size(-1) / num_head;
  if (!scale.isNone()) {
    return sd_flash_mha(
        query, key, value, num_head, headSize, scale.toDouble());
  } else {
    auto scale_ = 1.f / sqrt(headSize);
    return sd_flash_mha(query, key, value, num_head, headSize, scale_);
  }
}

template <typename T>
std::vector<at::Tensor> dil_mat_split(
    const at::Tensor& mat,
    const at::IntArrayRef& split_list) {
  int64_t batchSize = mat.dim() > 2 ? mat.size(0) : 1;
  int64_t sequenceSize = mat.dim() > 2 ? mat.size(1) : mat.size(0);
  int64_t total_size = (mat.dim() > 2 ? mat.size(2) : mat.size(1));
  int64_t split_size = split_list.size();
  std::vector<at::Tensor> split_mat;
  for (int i = 0; i < split_size; ++i) {
    split_mat.push_back(
        mat.dim() > 2
            ? at::empty({batchSize, sequenceSize, split_list[i]}, mat.dtype())
            : at::empty({sequenceSize, split_list[i]}, mat.dtype()));
  }
  auto mat_ = mat.contiguous();
  T* src = mat_.data_ptr<T>();
  at::parallel_for(
      0, batchSize * sequenceSize, 1, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
          int64_t accum = 0;
          for (int j = 0; j < split_size; ++j) {
            memcpy(
                split_mat[j].data_ptr<T>() + i * split_list[j],
                src + i * total_size + accum,
                sizeof(T) * split_list[j]);
            accum += split_list[j];
          }
        }
      });

  return split_mat;
}

c10::List<at::Tensor> dil_split_tensor(
    const at::Tensor& mat,
    const at::IntArrayRef& split_list) {
  RECORD_FUNCTION("dil_split_tensor", c10::ArrayRef<c10::IValue>({}));
  return c10::List<at::Tensor>(dil_mat_split<at::BFloat16>(mat, split_list));
}
} // namespace cpu
} // namespace torch_ipex
